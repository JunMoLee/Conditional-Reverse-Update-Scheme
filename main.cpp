/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
*   
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
*   
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer. 
*   
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
*   
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen     Email: pchen72 at asu dot edu 
*                     
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <random>
#include <vector>
#include "Cell.h"
#include "Array.h"
#include "formula.h"
#include "NeuroSim.h"
#include "Param.h"
#include "IO.h"
#include "Train.h"
#include "Test.h"
#include "Mapping.h"
#include "Definition.h"

using namespace std;

int main() {

	gen.seed(0);	
	/* Load in MNIST data */
	ReadTrainingDataFromFile("patch60000_train.txt", "label60000_train.txt");
	ReadTestingDataFromFile("patch10000_test.txt", "label10000_test.txt");
	/* Initialization of synaptic array from input to hidden layer */
	//arrayIH->Initialization<IdealDevice>();
	arrayIH->Initialization<RealDevice>();
	//arrayIH->Initialization<MeasuredDevice>();
	//arrayIH->Initialization<SRAM>(param->numWeightBit);
	//arrayIH->Initialization<DigitalNVM>(param->numWeightBit,true);

	
	/* Initialization of synaptic array from hidden to output layer */
	//arrayHO->Initialization<IdealDevice>();
	arrayHO->Initialization<RealDevice>();
	//arrayHO->Initialization<MeasuredDevice>();
	//arrayHO->Initialization<SRAM>(param->numWeightBit);
	//arrayHO->Initialization<DigitalNVM>(param->numWeightBit,true);


	/* Initialization of NeuroSim synaptic cores */
	param->relaxArrayCellWidth = 0;
	NeuroSimSubArrayInitialize(subArrayIH, arrayIH, inputParameterIH, techIH, cellIH);
	param->relaxArrayCellWidth = 1;
	NeuroSimSubArrayInitialize(subArrayHO, arrayHO, inputParameterHO, techHO, cellHO);
	/* Calculate synaptic core area */
	NeuroSimSubArrayArea(subArrayIH);
	NeuroSimSubArrayArea(subArrayHO);
	
	/* Calculate synaptic core standby leakage power */
	NeuroSimSubArrayLeakagePower(subArrayIH);
	NeuroSimSubArrayLeakagePower(subArrayHO);
	
	/* Initialize the neuron peripheries */
	NeuroSimNeuronInitialize(subArrayIH, inputParameterIH, techIH, cellIH, adderIH, muxIH, muxDecoderIH, dffIH, subtractorIH);
	NeuroSimNeuronInitialize(subArrayHO, inputParameterHO, techHO, cellHO, adderHO, muxHO, muxDecoderHO, dffHO, subtractorHO);
	/* Calculate the area and standby leakage power of neuron peripheries below subArrayIH */
	double heightNeuronIH, widthNeuronIH;
	NeuroSimNeuronArea(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH, subtractorIH, &heightNeuronIH, &widthNeuronIH);
	double leakageNeuronIH = NeuroSimNeuronLeakagePower(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH, subtractorIH);
	/* Calculate the area and standby leakage power of neuron peripheries below subArrayHO */
	double heightNeuronHO, widthNeuronHO;
	NeuroSimNeuronArea(subArrayHO, adderHO, muxHO, muxDecoderHO, dffHO, subtractorHO, &heightNeuronHO, &widthNeuronHO);
	double leakageNeuronHO = NeuroSimNeuronLeakagePower(subArrayHO, adderHO, muxHO, muxDecoderHO, dffHO, subtractorHO);
	
	/* Print the area of synaptic core and neuron peripheries */
	double totalSubArrayArea = subArrayIH->usedArea + subArrayHO->usedArea;
	double totalNeuronAreaIH = adderIH.area + muxIH.area + muxDecoderIH.area + dffIH.area + subtractorIH.area;
	double totalNeuronAreaHO = adderHO.area + muxHO.area + muxDecoderHO.area + dffHO.area + subtractorHO.area;
	printf("Total SubArray (synaptic core) area=%.4e m^2\n", totalSubArrayArea);
	printf("Total Neuron (neuron peripheries) area=%.4e m^2\n", totalNeuronAreaIH + totalNeuronAreaHO);
	printf("Total area=%.4e m^2\n", totalSubArrayArea + totalNeuronAreaIH + totalNeuronAreaHO);

	/* Print the standby leakage power of synaptic core and neuron peripheries */
	printf("Leakage power of subArrayIH is : %.4e W\n", subArrayIH->leakage);
	printf("Leakage power of subArrayHO is : %.4e W\n", subArrayHO->leakage);
	printf("Leakage power of NeuronIH is : %.4e W\n", leakageNeuronIH);
	printf("Leakage power of NeuronHO is : %.4e W\n", leakageNeuronHO);
	printf("Total leakage power of subArray is : %.4e W\n", subArrayIH->leakage + subArrayHO->leakage);
	printf("Total leakage power of Neuron is : %.4e W\n", leakageNeuronIH + leakageNeuronHO);
	
	/* Initialize weights and map weights to conductances for hardware implementation */
	WeightInitialize();
	if (param->useHardwareInTraining) { WeightToConductance(); }

	srand(0);	// Pseudorandom number seed

		double NL_LTP_Gp = static_cast<RealDevice*>(arrayIH->cell[0][0])->NL_LTP_Gp;
	    double NL_LTD_Gp = static_cast<RealDevice*>(arrayIH->cell[0][0])->NL_LTD_Gp;
		double NL_LTP_Gn = static_cast<RealDevice*>(arrayIH->cell[0][0])->NL_LTP_Gn;
	    double NL_LTD_Gn = static_cast<RealDevice*>(arrayIH->cell[0][0])->NL_LTD_Gn;
		double LA = param->alpha1;
		int reverseperiod = param->newUpdateRate;
		int refperiod = param->RefPeriod;
		int reverseupdate = param->ReverseUpdate;
		int kp = static_cast<RealDevice*>(arrayIH->cell[0][0])->maxNumLevelpLTP;
		int kd = static_cast<RealDevice*>(arrayIH->cell[0][0])->maxNumLevelpLTD;
		int knp = static_cast<RealDevice*>(arrayIH->cell[0][0])->maxNumLevelnLTP;
		int knd = static_cast<RealDevice*>(arrayIH->cell[0][0])->maxNumLevelnLTD;
		double pof = static_cast<RealDevice*>(arrayIH->cell[0][0])->pmaxConductance/static_cast<RealDevice*>(arrayIH->cell[0][0])->pminConductance;
		double nof = static_cast<RealDevice*>(arrayIH->cell[0][0])->nmaxConductance/static_cast<RealDevice*>(arrayIH->cell[0][0])->nminConductance;
		int fullrefresh = param ->FullRefresh;
		int refreshperiod = param -> RefreshRate;
		double Gth1 = param -> Gth1;
	  	double Gth2 = param -> Gth2;
		double revlr = LA / param -> ratio ;	
		int Reference = param -> Reference;
		double maxaccuracy=0;
		

														               
		printf("opt: %s  NL_LTP_Gp:%.1f NL_LTD_Gp:%.1f NL_LTP_Gn:%.1f NL_LTD_Gn:%.1f CSpP: %d CSpD: %d CSnP: %d CSnD: %d normal LR %.2f reverse LR %.2f\n", param->optimization_type, NL_LTP_Gp, NL_LTD_Gp, NL_LTP_Gn, NL_LTD_Gn, kp, kd, knp, knd, LA, revlr);
		printf("reverseupdate Y/N: %d refresh Y/N: %d reference Y/N : %d reverseperiod: %d refreshperiod: %d Gth1: %.1f Gth2: %.1f refperiod: %d\n", reverseupdate, fullrefresh, Reference, reverseperiod, refreshperiod, Gth1, Gth2, refperiod);
		bool write_or_not=1;
		fstream read;
		char str[1024];
		sprintf(str, "NL_%.2f_%.2f_Gth_%.2f_LR_%.2f_revLR_%.2f_%d_%d_numLevel_%d_cratio_%.2f_NLdrift_%.3f.csv" ,NL_LTP_Gp, NL_LTD_Gp, Gth1, LA, revlr, reverseperiod, refperiod, param->numInputLevel, param->cratio, param->NL_drift);
		read.open(str,fstream::app);                                                         
																	
		for (int i=1; i<=param->totalNumEpochs; i++) {


		cout << "Training Epoch : " << i << endl; 
		Train(i, param->numTrainImagesPerEpoch, param->interNumEpochs,param->optimization_type);
		if (!param->useHardwareInTraining && param->useHardwareInTestingFF) { WeightToConductance(); }
		Validate();
		// maximum accuracy 
		if (i==1)
		maxaccuracy=(double)correct/param->numMnistTestImages*100;
		else
		{
			if ((double)correct/param->numMnistTestImages*100 > maxaccuracy)
			{
				maxaccuracy=(double)correct/param->numMnistTestImages*100;
			}
		}

		if(write_or_not){

		read <<param->optimization_type<<", "<<NL_LTP_Gp<<", "<<NL_LTD_Gp<<", "<<NL_LTP_Gn<<", "<<NL_LTD_Gn<<", "<<kp<<", "<<kd<<", "<<knp<<", "<<knd<<", "<<LA<<", "<<revlr<<", " <<reverseupdate<<", "<<reverseperiod<<", "<<refperiod<<", "<<fullrefresh<<", "<<refreshperiod<<", "<<i*param->interNumEpochs<< ", "<<param->errorcount<<", "<<(double)correct/param->numMnistTestImages*100 <<", "<<  maxaccuracy<< endl;
		
		}
		printf("%.2f, max: %.2f, errorcount: %.2f, NL drift: %.3f \n", (double)correct/param->numMnistTestImages*100, maxaccuracy, param->errorcount,(param->epoch_cell-1)*param->NL_drift/(param->totalNumEpochs-1) );
		
		/*printf("\tRead latency=%.4e s\n", subArrayIH->readLatency + subArrayHO->readLatency);
		printf("\tWrite latency=%.4e s\n", subArrayIH->writeLatency + subArrayHO->writeLatency);
		printf("\tRead energy=%.4e J\n", arrayIH->readEnergy + subArrayIH->readDynamicEnergy + arrayHO->readEnergy + subArrayHO->readDynamicEnergy);
		printf("\tWrite energy=%.4e J\n", arrayIH->writeEnergy + subArrayIH->writeDynamicEnergy + arrayHO->writeEnergy + subArrayHO->writeDynamicEnergy);*/
	}
	printf("\n");
        printf("\n");

	return 0;
}




