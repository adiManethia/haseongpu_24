#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <write_to_vtk.h>
#include <calc_phi_ase.h>
#include <map_rays_to_prisms.h>
#include <cudachecks.h>
#include <importance_sampling.h>
#include <calc_sample_phi_ase.h>
#include <mesh.h>
#include <progressbar.h> /*progressBar */
#include <logging.h>
#include <types.h>

#define SEED 4321
#define BEST_MSE 0
#define BEST_ASE 1
#define BEST_RAYNUMBER 2

double calcMSE(const double phiAse, const double phiAseSquare, const unsigned raysPerSample){
  double a = phiAseSquare / raysPerSample;
  double b = (phiAse / raysPerSample) * (phiAse / raysPerSample);

  return sqrt(abs((a - b) / raysPerSample));
}


float calcPhiAse ( unsigned hRaysPerSample,
    const unsigned maxRaysPerSample,
    const unsigned maxRepetitions,
    const Mesh& dMesh,
    const Mesh& hMesh,
    const std::vector<double>& hSigmaA,
    const std::vector<double>& hSigmaE,
    const std::vector<float>& mseThreshold,
    const bool useReflections,
    std::vector<float> &phiAse,
    std::vector<double> &mse,
    std::vector<unsigned> &totalRays,
    unsigned gpu_i,
    unsigned minSample_i,
    unsigned maxSample_i,
    float &runtime){

  // Optimization to use more L1 cache
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  cudaSetDevice(gpu_i);

  using thrust::device_vector;
  using thrust::raw_pointer_cast;

  // variable Definitions CPU
  time_t starttime                = time(0);
  unsigned hRaysPerSampleSave     = hRaysPerSample;
  unsigned maxReflections         = useReflections ? hMesh.getMaxReflections() : 0;
  unsigned reflectionSlices       = 1 + (2 * maxReflections);
  unsigned numberOfWavelengths    = hSigmaE.size();
  // In some cases distributeRandomly has to be true !
  // Otherwise bad or no ray distribution possible.
  bool distributeRandomly         = true;
  dim3 blockDim(128);             //can't be more than 256 due to restrictions from the Mersenne Twister
  dim3 gridDim(200);              //can't be more than 200 due to restrictions from the Mersenne Twister

  // Memory allocation/init and copy for device memory
  device_vector<unsigned> dNumberOfReflections(maxRaysPerSample, 0);
  device_vector<float>    dGainSum            (1, 0);
  device_vector<float>    dGainSumSquare      (1, 0);
  device_vector<unsigned> dLostRays           (1, 0);
  device_vector<unsigned> dRaysPerPrism       (hMesh.numberOfPrisms * reflectionSlices, 1);
  device_vector<unsigned> dPrefixSum          (hMesh.numberOfPrisms * reflectionSlices, 0);
  device_vector<double>   dImportance         (hMesh.numberOfPrisms * reflectionSlices, 0);
  device_vector<double>   dImportanceSave     (hMesh.numberOfPrisms * reflectionSlices, 0);
  device_vector<unsigned> dIndicesOfPrisms    (maxRaysPerSample,  0);

  // OUTPUT DATA
  // thrust::host_vector<unsigned> hNumberOfReflections(maxRaysPerSample,0);
  // thrust::host_vector<unsigned> hIndicesOfPrisms(maxRaysPerSample,0);
  //thrust::host_vector<unsigned> hRaysPerPrism(hMesh.numberOfPrisms * reflectionSlices, 0);
  //unsigned midRaysPerSample=0;

  // CUDA Mersenne twister (can not have more than 200 blocks!)
  curandStateMtgp32 *devMTGPStates;
  mtgp32_kernel_params *devKernelParams;
  CUDA_CALL(cudaMalloc((void **)&devMTGPStates, gridDim.x  * sizeof(curandStateMtgp32)));
  CUDA_CALL(cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params)));
  CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams));
  CURAND_CALL(curandMakeMTGP32KernelState(devMTGPStates, mtgp32dc_params_fast_11213, devKernelParams, gridDim.x, SEED + minSample_i));

  // Calculate Phi Ase for each wavelength
  for(unsigned wave_i = 0; wave_i < numberOfWavelengths; ++wave_i){

    // Calculation for each sample point
    for(unsigned sample_i = minSample_i; sample_i < maxSample_i; ++sample_i){
      std::vector<float> bestASE(1000,3);
      //unsigned sample_i = 1;{
      unsigned sampleOffset  = sample_i + hMesh.numberOfSamples * wave_i;
      unsigned hRaysPerSampleDump = 0; 
      hRaysPerSample = hRaysPerSampleSave;
      bool mseTooHigh=true;

      float hSumPhi =  importanceSamplingPropagation(
          sample_i,
          reflectionSlices,
          dMesh,
          hMesh.numberOfPrisms,
          hSigmaA[wave_i],
          hSigmaE[wave_i],
          dImportanceSave);
      while(mseTooHigh){
        CURAND_CALL(curandMakeMTGP32KernelState(devMTGPStates, mtgp32dc_params_fast_11213, devKernelParams, gridDim.x, SEED + sample_i));
        unsigned run = 0;
        while(run < maxRepetitions && mseTooHigh){
          run++;
	  dLostRays[0] = 0;

          thrust::copy(dImportanceSave.begin(),dImportanceSave.end(),dImportance.begin());
          hRaysPerSampleDump = importanceSamplingDistribution(
              reflectionSlices,
              dMesh,
              hMesh.numberOfPrisms,
              hRaysPerSample,
              dImportance, 
              dRaysPerPrism,
              hSumPhi,
              distributeRandomly);
	  // DEBUG
          // if(dRaysPerPrism[6495] > 10000){
          //   dout(V_DEBUG) << "Too high raysPerprism " << dRaysPerPrism[6495] << " sample_i: " << sample_i <<std::endl;
          //   exit(0);
          // }

          // Prism scheduling for gpu threads
          mapRaysToPrisms(dIndicesOfPrisms, dNumberOfReflections, dRaysPerPrism, dPrefixSum, reflectionSlices, hRaysPerSampleDump, hMesh.numberOfPrisms);

          // DEBUG
          // if(sample_i == 0){
          //   thrust::copy(dNumberOfReflections.begin(),dNumberOfReflections.end(),hNumberOfReflections.begin());
          //   thrust::copy(dIndicesOfPrisms.begin(),dIndicesOfPrisms.end(),hIndicesOfPrisms.begin());
          // thrust::copy(dRaysPerPrism.begin(), dRaysPerPrism.end(), hRaysPerPrism.begin());
          //   midRaysPerSample=hRaysPerSample;
          // }

          // Start Kernel
          dGainSum[0]       = 0;
          dGainSumSquare[0] = 0;

          if(useReflections){
            calcSampleGainSum<<< gridDim, blockDim >>>( devMTGPStates,
                dMesh, 
                raw_pointer_cast(&dIndicesOfPrisms[0]), 
                wave_i, 
                raw_pointer_cast(&dNumberOfReflections[0]), 
                raw_pointer_cast(&dImportance[0]),
                hRaysPerSampleDump, 
                raw_pointer_cast(&dGainSum[0]), 
                raw_pointer_cast(&dGainSumSquare[0]),
                raw_pointer_cast(&dLostRays[0]),
                sample_i, 
                hSigmaA[wave_i], 
                hSigmaE[wave_i],
                raw_pointer_cast(&(device_vector<unsigned> (1,0))[0]));
          }
          else{
            calcSampleGainSumWithoutReflections<<< gridDim, blockDim >>>( devMTGPStates,
                dMesh, 
                raw_pointer_cast(&dIndicesOfPrisms[0]), 
                wave_i, 
                raw_pointer_cast(&dImportance[0]),
                hRaysPerSampleDump, 
                raw_pointer_cast(&dGainSum[0]), 
                raw_pointer_cast(&dGainSumSquare[0]),
                sample_i, 
                hSigmaA[wave_i], 
                hSigmaE[wave_i],
                raw_pointer_cast(&(device_vector<unsigned> (1,0))[0]));
          }

	  // Remove lost rays (reflections) from ray counter
	  hRaysPerSampleDump -= dLostRays[0];

          float mseTmp = calcMSE(dGainSum[0], dGainSumSquare[0], hRaysPerSampleDump);

          // DEBUG
          if(isnan(mseTmp)){
            dout(V_ERROR) << "mseTmp: " << mseTmp << " gainSum:" << dGainSum[0] << " gainSum²:" << dGainSumSquare[0] << " RaysPerSample:" << hRaysPerSampleDump <<std::endl;
          }

          assert(!isnan(dGainSum[0]));
          assert(!isnan(dGainSumSquare[0]));
          assert(!isnan(mseTmp));

          //MSE TESTs
          // if(mseTmp > mse.at(sampleOffset)){
          //   // this happens in calcMSE
          //   double ca = dGainSumSquare[sampleOffset] / hRaysPerSampleDump;
          //   double cb = (dGainSum[sampleOffset] / hRaysPerSampleDump) * (dGainSum[sampleOffset] / hRaysPerSampleDump);

          //   dout(V_WARNING) << "MSE_BUG for sample " << sample_i << ": " << mseTmp << " > " << mse.at(sampleOffset) << std::endl;
          //   dout(V_DEBUG) << "Run: " << run << std::endl;
          //   dout(V_DEBUG) << "RaysPerSample: " << hRaysPerSample << std::endl;
          //   dout(V_DEBUG) << "RaysPerSampleDump: "<< hRaysPerSampleDump << std::endl;
          //   dout(V_DEBUG) << "phiAseSquare / raysPerSample = " << ca << std::endl; 
          //   dout(V_DEBUG) << "(phiAse / raysPerSample) * (phiAse / raysPerSample) = " << cb << std::endl; 
          //   dout(V_DEBUG) << "sqrt(abs((a - b) / raysPerSample)) = " << sqrt(abs((ca - cb) / hRaysPerSampleDump)) << std::endl; 
          //   dout(V_DEBUG) << std::endl;
          // }

          mse.at(sampleOffset) = mseTmp;
          if(mse.at(sampleOffset) < bestASE[BEST_MSE]){
            bestASE[BEST_MSE] = mse.at(sampleOffset);
            bestASE[BEST_ASE] = dGainSum[0];
            bestASE[BEST_RAYNUMBER] = hRaysPerSampleDump;
          }
          if(mse.at(sampleOffset) < mseThreshold.at(wave_i)) mseTooHigh = false;
        }

        // If the threshold is still too high, increase the number of rays and reset the previously calculated value
        if(hRaysPerSample * RAY_MULTIPLICATOR > (unsigned long)maxRaysPerSample) break;
        hRaysPerSample             *= RAY_MULTIPLICATOR;

      }
      // Update progressbar
      fancyProgressBar(maxSample_i);

      // get phiASE
      //phiAse.at(sampleOffset) = dGainSum[0];
      //phiAse.at(sampleOffset)   /= hRaysPerSampleDump * 4.0f * M_PI;
      //totalRays.at(sampleOffset)  = hRaysPerSampleDump;

      phiAse.at(sampleOffset) = bestASE[BEST_ASE];
      phiAse.at(sampleOffset)   /= bestASE[BEST_RAYNUMBER] * 4.0f * M_PI;
      totalRays.at(sampleOffset)  = bestASE[BEST_RAYNUMBER];

    }

    }

    // JUST OUTPUT
    // std::vector<unsigned> reflectionsPerPrism(hMesh.numberOfPrisms, 0);
    // std::vector<unsigned> raysPerPrism(hMesh.numberOfPrisms, 0);

    // for(unsigned i=0; i < midRaysPerSample; ++i){
    //   unsigned index = hIndicesOfPrisms[i];
    //   reflectionsPerPrism[index] = max(reflectionsPerPrism[index], (hNumberOfReflections[i] + 1) / 2);
    // }

    // for(unsigned i=0; i < hMesh.numberOfPrisms; ++i){
    //   for(unsigned j=0; j < reflectionSlices; ++j){
    //     unsigned index = i + hMesh.numberOfPrisms * j;
    //     raysPerPrism[i] += hRaysPerPrism[index];
    //   }
    // }

    //writePrismToVtk(hMesh, reflectionsPerPrism, "octrace_0_reflections", hRaysPerSample, maxRaysPerSample, mseThreshold.at(0), useReflections, 0);
    //writePrismToVtk(hMesh, raysPerPrism, "octrace_0_rays", hRaysPerSample, maxRaysPerSample, mseThreshold.at(0), useReflections, 0);

    //dout(V_INFO | V_NOLABEL) << "\n" << std::endl;
    // Free Memory
    cudaFree(devMTGPStates);
    cudaFree(devKernelParams);


    runtime = difftime(time(0),starttime);
    return runtime;
  }
