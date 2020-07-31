

// conv 3x3 for group (depth-wise convolutions)

#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include "summernet_hls.h"


void load_weights(FIX_WT weight_buf[16],
				  FIX_WT weights[16][3][3],
				  int i, int j)
{
#pragma HLS ARRAY_PARTITION variable=weights dim=1 factor=16

	for(int coo = 0; coo < 16; coo++){
#pragma HLS unroll
		weight_buf[coo] = weights[coo][i][j];
		
	}
}


void CONV_3x3_group(FIX_FM bottom[16][22][42],
					FIX_FM top[16][22][42],
					FIX_WT weights[16][3][3])
{

	FIX_WT weight_buf[16];

#pragma HLS ARRAY_PARTITION variable=bottom cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=top cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=weight_buf complete


	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){

#pragma HLS dataflow

			load_weights(weight_buf, weights, i, j);

			for(int h = 1; h <= 20; h++){
				for(int w = 1; w <= 40; w++){
#pragma HLS pipeline
					for(int co = 0; co < 16; co++){
#pragma HLS unroll
						
						top[co][h][w] += weight_buf[co] * bottom[co][h+i-1][w+j-1];
					}
				}
			}
		}
	}

}

// Conv 1x1 PE

FIX_32_12 compute_engine_16(FIX_WT w0,  FIX_FM b0,
					  FIX_WT w1,  FIX_FM b1,
					  FIX_WT w2,  FIX_FM b2,
					  FIX_WT w3,  FIX_FM b3,
					  FIX_WT w4,  FIX_FM b4,
					  FIX_WT w5,  FIX_FM b5,
					  FIX_WT w6,  FIX_FM b6,
					  FIX_WT w7,  FIX_FM b7,
					  FIX_WT w8,  FIX_FM b8,
					  FIX_WT w9,  FIX_FM b9,
					  FIX_WT w10, FIX_FM b10,
					  FIX_WT w11, FIX_FM b11,
					  FIX_WT w12, FIX_FM b12,
					  FIX_WT w13, FIX_FM b13,
					  FIX_WT w14, FIX_FM b14,
					  FIX_WT w15, FIX_FM b15)
{
	FIX_32_12 mul0, mul1, mul2,  mul3,  mul4,  mul5,  mul6,  mul7;
	FIX_32_12 mul8, mul9, mul10, mul11, mul12, mul13, mul14, mul15;
	FIX_32_12 add0, add1, add2, add3,  add4,  add5,  add6;
	FIX_32_12 add7, add8, add9, add10, add11, add12, add13, add14;

	mul0  = w0  * b0;
	mul1  = w1  * b1;
	mul2  = w2  * b2;
	mul3  = w3  * b3;
	mul4  = w4  * b4;
	mul5  = w5  * b5;
	mul6  = w6  * b6;
	mul7  = w7  * b7;
	mul8  = w8  * b8;
	mul9  = w9  * b9;
	mul10 = w10 * b10;
	mul11 = w11 * b11;
	mul12 = w12 * b12;
	mul13 = w13 * b13;
	mul14 = w14 * b14;
	mul15 = w15 * b15;


	add0 = mul0  + mul1;
	add1 = mul2  + mul3;
	add2 = mul4  + mul5;
	add3 = mul6  + mul7;
	add4 = mul8  + mul9;
	add5 = mul10 + mul11;
	add6 = mul12 + mul13;
	add7 = mul14 + mul15;

	add8  = add0 + add1;
	add9  = add2 + add3;
	add10 = add4 + add5;
	add11 = add6 + add7;

	add12 = add8  + add9;
	add13 = add10 + add11;

	add14 = add12 + add13;

	return add14;

}






void CONV_1x1(FIX_FM bottom[16][22][42],
			  FIX_FM top[16][22][42],
			  FIX_WT weights[16][16])
{
FIX_WT weight_buf[16][16];
FIX_32_12 tmp[16];

#pragma HLS ARRAY_PARTITION variable=bottom cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=top cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=weight_buf dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=weight_buf dim=2 factor=16
#pragma HLS ARRAY_PARTITION variable=tmp complete

#pragma HLS ALLOCATION instances=compute_engine_16 limit=9 function

	for(int i = 0; i < 16; i++)
		for(int j = 0; j < 16; j++)
			weight_buf[i][j] = weights[i][j];


	for(int h = 1; h <= 20; h++){
		for(int w = 1; w <= 40; w++) {


#pragma HLS pipeline

				for(int coo = 0; coo < 16; coo++) {
#pragma HLS unroll

					tmp[coo] = compute_engine_16(
												 weight_buf[coo][0],  bottom[0][h][w],
												 weight_buf[coo][1],  bottom[1][h][w],
												 weight_buf[coo][2],  bottom[2][h][w],
												 weight_buf[coo][3],  bottom[3][h][w],
												 weight_buf[coo][4],  bottom[4][h][w],
												 weight_buf[coo][5],  bottom[5][h][w],
												 weight_buf[coo][6],  bottom[6][h][w],
												 weight_buf[coo][7],  bottom[7][h][w],
												 weight_buf[coo][8],  bottom[8][h][w],
												 weight_buf[coo][9],  bottom[9][h][w],
												 weight_buf[coo][10], bottom[10][h][w],
												 weight_buf[coo][11], bottom[11][h][w],
												 weight_buf[coo][12], bottom[12][h][w],
												 weight_buf[coo][13], bottom[13][h][w],
												 weight_buf[coo][14], bottom[14][h][w],
												 weight_buf[coo][15], bottom[15][h][w]);
				}

				for(int coo = 0; coo < 16; coo++)
#pragma HLS unroll
					top[coo][h][w] += tmp[coo];

//			}
		}
	}
}
