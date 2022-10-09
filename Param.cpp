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

#include <string>
#include "math.h"
#include "Param.h"

Param::Param() {

	/* MNIST dataset */

	numMnistTrainImages = 60000;// # of training images in MNIST
	numMnistTestImages = 10000;	// # of testing images in MNIST

	/* Algorithm parameters */

	numTrainImagesPerEpoch = 8000;	// # of training images per epoch
	totalNumEpochs = 100;	// Total number of epochs
	interNumEpochs = 1;		// Internal number of epochs (print out the results every interNumEpochs)
	nInput = 400;     // # of neurons in input layer
	nHide = 100;      // # of neurons in hidden layer
	nOutput = 10;     // # of neurons in output layer
	maxWeight = 1;	// Upper bound of weight value
	minWeight = -1;	// Lower bound of weight value

    /*Optimization method 

    Available option include: "SGD", "Momentum", "Adagrad", "RMSprop" and "Adam"*/

    optimization_type = "SGD";

	/* nonlinearity setting */ 

	param_gp=1; // LTP nonlinearity
	param_gn=-9; // LTD nonlinearity

	/* weight, synapse distribution record */

	RefPeriod = 100;
	Record = 0;
	RecordPeriod = 200;
	WeightTrack = 0;
	WeightTrackPeriod=200;
	LocationTrack=0;
	LocationTrackPeriod=200;
	WeightDistribution=1;

	/* c2c, reference, refresh, reverse update period */

	cratio=0; // for write variation
	c2cvariation=0; // c2c variation turn on/off
    Reference = 0; // reference turn on/off (turn on:1, turon off:0, the same applies for below)
	FullRefresh= 0; // refresh function turn on/off
	RefreshRate = 2; // refresh period
	ReverseUpdate = 1; // reverse update function turn on/off
	newUpdateRate = 2; // reverse update period

	Gth1 = -1; // Gth for optimization
	Gth2 = 9; // not used
	TP_FN_record=0; // TP/FN  (at Test.cpp)

	/* Hardware parameters */

	useHardwareInTrainingFF = true;   // Use hardware in the feed forward part of training or not (true: realistic hardware, false: ideal software)
	useHardwareInTrainingWU = true;   // Use hardware in the weight update part of training or not (true: realistic hardware, false: ideal software)
	useHardwareInTraining = useHardwareInTrainingFF || useHardwareInTrainingWU;    // Use hardware in the training or not
	useHardwareInTestingFF = true;    // Use hardware in the feed forward part of testing or not (true: realistic hardware, false: ideal software)
	numBitInput = 1;       // # of bits of the input data (=1 for black and white data)
	numBitPartialSum = 8;  // # of bits of the digital output (partial weighted sum output)
	pSumMaxHardware = pow(2, numBitPartialSum) - 1;   // Max digital output value of partial weighted sum
	numInputLevel = pow(2, numBitInput);  // # of levels of the input data
	numWeightBit = 6;	// # of weight bits (only for pure algorithm, SRAM and digital RRAM hardware)
	BWthreshold = 0.5;	// The black and white threshold for numBitInput=1
	Hthreshold = 0.5;	// The spiking threshold for the hidden layer (da1 in Train.cpp and Test.cpp)
	numColMuxed = 16;	// How many columns share 1 read circuit (for analog RRAM) or 1 S/A (for digital RRAM)
	numWriteColMuxed = 16;	// How many columns share 1 write column decoder driver (for digital RRAM)
	writeEnergyReport = true;	// Report write energy calculation or not
	NeuroSimDynamicPerformance = true; // Report the dynamic performance (latency and energy) in NeuroSim or not
	relaxArrayCellHeight = 0;	// True: relax the array cell height to standard logic cell height in the synaptic array
	relaxArrayCellWidth = 0;	// True: relax the array cell width to standard logic cell width in the synaptic array
	arrayWireWidth = 100;	// Array wire width (nm)
	processNode = 32;	// Technology node (nm)
	clkFreq = 2e9;		// Clock frequency (Hz)


// additional code for automization using bash script, not necessary
const int
a=3;


// Settings tested in paper - uncomment to apply the settings
// Optimal Gth values earned from simulation {NL(LTP), optimal Gth from grid search over 0.2-9.8}

/* LR 0.15 */
/*
param_gn=-9;
double scan [8][2] = {{1,1},{2,1.2},{3,1.6},{4,2.4},{5,3.4},{6,3.8},{7,6.2},{8,8.6}};

const double // const define for automation
l=15;
alpha1 =l/100.0;	// Learning rate for the weights from input to hidden layer
alpha2 =l/2.0/100.0;	// Learning rate for the weights from hidden to output layer

// define lr during reverse update
const double
revlr=15;
ratio = alpha1 / (revlr/100);
*/

/* LR 0.1 */
/*
double scan [8][2] = {{1,0.6},{2,0.8},{3,1},{4,1.2},{5,2.2},{6,2.4},{7,3.8},{8,6.4}};
param_gn=-9;

const double // const define for automation
l=10;
alpha1 =l/100.0;	// Learning rate for the weights from input to hidden layer
alpha2 =l/2.0/100.0;	// Learning rate for the weights from hidden to output layer

// define lr during reverse update
const double
revlr=15;
ratio = alpha1 / (revlr/100);
*/

/* LTD=-8 LR=0.1 */
/*
//param_gn=-8;
//double scan [8][2] = {{1,0.8},{2,1},{3,1.6},{4,2.2},{5,2.6},{6,3.4},{7,5},{8,6.8}};
param_gn=-9;

const double // const define for automation
l=10;
alpha1 =l/100.0;	// Learning rate for the weights from input to hidden layer
alpha2 =l/2.0/100.0;	// Learning rate for the weights from hidden to output layer

// define lr during reverse update
const double
revlr=15;
ratio = alpha1 / (revlr/100);
*/

/* RUP NL 1, 5 LR 0.1  */ 
// {RUP, optimal Gth from grid search over 0.2-9.8, NL(LTP)}
//param_gn=-8;
//double scan[6][3] = {{3,4.4,5},{5,6.2,5},{10,8.4,5},{3,0.8,1},{5,1.4,1},{10,3.2,1}};

/* C2C parameterlist */
// NL(NL level) alpha1, Optimized Gth1 - please apply these settings
// NL1 0.3, 2.4 / NL3 0.3, 3.8 / NL5 0.1, 2.2 / NL8 0.1, 6.4 

const double // const define for automation
l=10; // (30 for NL1, NL3. 10 for NL5, NL8)
alpha1 =l/100.0;	// Learning rate for the weights from input to hidden layer
alpha2 =l/2.0/100.0;	// Learning rate for the weights from hidden to output layer
double scan[4][2]={{1,2.4},{3, 3.8},{5, 2.2},{8, 6.4}};
	
// define lr during reverse update
const double
revlr=15;
ratio = alpha1 / (revlr/100);
	
////////// Common parameters //////////
// NL drift (added)
const double
nd=10
;
NL_drift=nd/10;
use_drift=1;
	
const double // const define for c2c variation
cratioo=50;

c2cvariation=1; // c2c variation turn on/off
cratio=cratioo; // for write variation
numBitInput = 1; // # of bits of the input data (=1 for black and white data)
Reference=0; // reference turn on and off
RefPeriod=1; // reference period
ReverseUpdate = 1; // reverse update function turn on/off
newUpdateRate= 2; // reverse update date
FullRefresh= 0; // refresh function turn on/off
RefreshRate = 2; // refresh perio

/// Nonlinearity dependent parameters ///
	
param_gp=scan[a][0];
Gth1=scan[a][1];

	
}


