#include <CL/cl.h>
#include "facegen.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
/*
 * TODO
 * Define global variables here. For example,
 * cl_platform_id platform;
 */
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;
cl_int err;

// feature maps buffer
cl_mem bfm0, bfm1, bfm2, bfm3, boutput;

// weight buffer
cl_mem btconv1_w, btconv2_w, btconv3_w, btconv4_w;

// bias buffer
cl_mem btconv1_b, btconv2_b, btconv3_b, btconv4_b;


#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

char *get_source_code(const char *file_name, size_t *len) {
    char *source_code;
    size_t length;
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    length = (size_t)ftell(file);
    rewind(file);

    source_code = (char *)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';

    fclose(file);

    *len = length;
    return source_code;
}

/*
 * linear projection (matrix-vector multiplication)
 * in : (C)
 * out : (K)
 * weight : (C, K)
 * bias : (K)
 */
static void proj(float *in, float *out, float *weight, float *bias, int C, int K) {
    for (int k = 0; k < K; ++k) {
        float s = 0;
        for (int c = 0; c < C; ++c) {
            s += in[c] * weight[c * K + k];
        }
        s += bias[k];
        out[k] = s;
    }
}

/*
 * batch normalization (in-place)
 * inout : (H, W, C)
 * beta : (C)
 * gamma : (C)
 * mean : (C)
 * var : (C)
 */
static void batch_norm(float *inout, float *beta, float *gamma, float *mean, float *var, int HW, int C) {
    for (int hw = 0; hw < HW; ++hw) {
        for (int c = 0; c < C; ++c) {
            float scaled_gamma = gamma[c] / sqrtf(var[c] + 1e-5);
            inout[hw * C + c] = scaled_gamma * inout[hw * C + c] + (beta[c] - scaled_gamma * mean[c]);
        }
    }
}

/*
 * ReLU (in-place)
 * inout : (H, W, C)
 */
static void relu(float *inout, int HWC) {
    for (int hwc = 0; hwc < HWC; ++hwc) {
        inout[hwc] = fmaxf(inout[hwc], 0);
    }
}

/*
 * transposed convolution
 * in : (H_IN, W_IN, C)
 * out : (H_IN * 2, W_IN * 2, K)
 * weight : (5, 5, K, C)
 * bias : (K)
 */
//static void tconv(float *in, float *out, float *weight, float *bias, int H_IN, int W_IN, int C, int K) {
//    int H_OUT = H_IN * 2, W_OUT = W_IN * 2;
//    for (int h_out = 0; h_out < H_OUT; ++h_out) {
//        for (int w_out = 0; w_out < W_OUT; ++w_out) {
//            for (int k = 0; k < K; ++k) {
//                float ss = 0;
//                for (int r = 0; r < 5; ++r) {
//                    for (int s = 0; s < 5; ++s) {
//                        // top and left side has padding 3, bottom and right side has padding 2
//                        // so subtract 3
//                        int h_in = h_out - 3 + r;
//                        int w_in = w_out - 3 + s;
//                        // stride is 2, so check coordinates fall into input element or empty space
//                        if (h_in % 2 == 0 && w_in % 2 == 0) {
//                            h_in /= 2;
//                            w_in /= 2;
//                            // boundary check
//                            if (0 <= h_in && h_in < H_IN && 0 <= w_in && w_in < W_IN) {
//                                for (int c = 0; c < C; ++c) {
//                                    // filter is stored in reverse; so use [4 - r][4 - s] instead of [r][s]
//                                    // ss += in[h_in][w_in][c] * weight[4 - r][4 - s][k][c];
//                                    ss += in[(h_in * W_IN + w_in) * C + c] * weight[(((4 - r) * 5 + (4 - s)) * K + k) * C + c];
//                                }
//                            }
//                        }
//                    }
//                }
//                ss += bias[k];
//                // out[h_out][w_out][k] = ss;
//                out[(h_out * W_OUT + w_out) * K + k] = ss;
//            }
//        }
//    }
//}

/*
 * tanh (in-place)
 * inout : (H, W, C)
 */
static void tanh_layer(float *inout, int HWC) {
    for (int hwc = 0; hwc < HWC; ++hwc) {
        inout[hwc] = tanhf(inout[hwc]);
    }
}



void facegen_init() {
    /*
     * TODO
     * Initialize OpenCL objects as global variables. For example,
     * clGetPlatformIDs(1, &platform, NULL);
     */
    // Variables for kernelargs
    size_t source_size;
    char *source_code;

    //get platform id
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    //get Device id
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    //Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    //Create commandqueue
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);

    //Create object
    source_code = get_source_code("kernel.cl", &source_size);
    program = clCreateProgramWithSource(context, 1,
            (const char**)&source_code, &source_size, &err);
    CHECK_ERROR(err);

    //program build
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if(err == CL_BUILD_PROGRAM_FAILURE){
        size_t log_size;
        char *log;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        log = (char *)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        printf("Compile error:\n%s\n", log);
        free(log);

        exit(0);
    }

    //create feature maps buffer
    bfm0 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 4 * 4 * 512, NULL, &err);
    CHECK_ERROR(err);
    bfm1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 8 * 8 * 256, NULL, &err);
    CHECK_ERROR(err);
    bfm2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 16 * 16 * 128, NULL, &err);
    CHECK_ERROR(err);
    bfm3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 64, NULL, &err);
    CHECK_ERROR(err);
    boutput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 64 * 64 * 3, NULL, &err);
    CHECK_ERROR(err);

    //create weight buffer
    btconv1_w = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 5 * 5 * 256 * 512, NULL, &err);
    CHECK_ERROR(err);
    btconv2_w  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 5 * 5 * 128 * 256, NULL, &err);
    CHECK_ERROR(err);
    btconv3_w  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 5 * 5 * 64 * 128, NULL, &err);
    CHECK_ERROR(err);
    btconv4_w  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 5 * 5 * 3 * 64, NULL, &err);
    CHECK_ERROR(err);

    //create bias buffer
    btconv1_b = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 256, NULL, &err);
    CHECK_ERROR(err);
    btconv2_b  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 128, NULL, &err);
    CHECK_ERROR(err);
    btconv3_b  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 64, NULL, &err);
    CHECK_ERROR(err);
    btconv4_b  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3, NULL, &err);
    CHECK_ERROR(err);

}

void facegen(int num_to_gen, float *network, float *inputs, float *outputs) {
    int w_in, h_in, C, K;

    // split network into each layer's parameter
    float *proj_w = network; network += 100 * 8192;
    float *proj_b = network; network += 8192;
    float *bn0_beta = network; network += 512;
    float *bn0_gamma = network; network += 512;
    float *bn0_mean = network; network += 512;
    float *bn0_var = network; network += 512;
    float *tconv1_w = network; network += 5 * 5 * 256 * 512;
    float *tconv1_b = network; network += 256;
    float *bn1_beta = network; network += 256;
    float *bn1_gamma = network; network += 256;
    float *bn1_mean = network; network += 256;
    float *bn1_var = network; network += 256;
    float *tconv2_w = network; network += 5 * 5 * 128 * 256;
    float *tconv2_b = network; network += 128;
    float *bn2_beta = network; network += 128;
    float *bn2_gamma = network; network += 128;
    float *bn2_mean = network; network += 128;
    float *bn2_var = network; network += 128;
    float *tconv3_w = network; network += 5 * 5 * 64 * 128;
    float *tconv3_b = network; network += 64;
    float *bn3_beta = network; network += 64;
    float *bn3_gamma = network; network += 64;
    float *bn3_mean = network; network += 64;
    float *bn3_var = network; network += 64;
    float *tconv4_w = network; network += 5 * 5 * 3 * 64;
    float *tconv4_b = network; network += 3;

    // intermediate buffer for feature maps
    float *fm0 = (float*)malloc(4 * 4 * 512 * sizeof(float));
    float *fm1 = (float*)malloc(8 * 8 * 256 * sizeof(float));
    float *fm2 = (float*)malloc(16 * 16 * 128 * sizeof(float));
    float *fm3 = (float*)malloc(32 * 32 * 64 * sizeof(float));

    // Work_items and work_group
    size_t global_size[3], local_size[3];

    //Write each stage buffers
    //feature map 0
    w_in = 4; h_in = 4; C = 512; K = 256;
    err = clEnqueueWriteBuffer(queue, btconv1_w, CL_FALSE, 0, sizeof(float) * 5 * 5 * C * K, tconv1_w, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, btconv1_b, CL_TRUE, 0, sizeof(float) * K, tconv1_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    //feature map 1
    w_in *= 2; h_in *= 2; C /= 2; K /= 2;
    err = clEnqueueWriteBuffer(queue, btconv2_w, CL_FALSE, 0, sizeof(float) * 5 * 5 * C * K, tconv2_w, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, btconv2_b, CL_TRUE, 0, sizeof(float) * K, tconv2_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    //feature map 2
    w_in *= 2; h_in *= 2; C /= 2; K /= 2;
    err = clEnqueueWriteBuffer(queue, btconv3_w, CL_FALSE, 0, sizeof(float) * 5 * 5 * C * K, tconv3_w, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, btconv3_b, CL_TRUE, 0, sizeof(float) * K, tconv3_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    //feature map 3
    w_in *= 2; h_in *= 2; C /= 2; K = 3;
    err = clEnqueueWriteBuffer(queue, btconv4_w, CL_FALSE, 0, sizeof(float) * 5 * 5 * C * K, tconv4_w, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, btconv4_b, CL_TRUE, 0, sizeof(float) * K, tconv4_b, 0, NULL, NULL);
    CHECK_ERROR(err);


    // create kernel tconv
    kernel = clCreateKernel(program, "tconv_k", &err);
    CHECK_ERROR(err);

    //time result array
    double result_time[14], time_start, time_end;
    int loop;
    for(loop = 0; loop < 14; loop++){
        result_time[loop] = 0;
    }

    // run network for each face
    for (int n = 0; n < num_to_gen; ++n) {
        float *input = inputs + n * 100;
        float *output = outputs + n * 64 * 64 * 3;

        //Input image
        time_start = get_time();
        proj(input, fm0, proj_w, proj_b, 100, 8192);
        time_end = get_time();
        result_time[0] += time_end - time_start;
        // implicit layout change here; (8192,) -> (4, 4, 512)

        time_start = get_time();
        batch_norm(fm0, bn0_beta, bn0_gamma, bn0_mean, bn0_var, 4 * 4, 512);
        time_end  = get_time();
        result_time[1] += time_end - time_start;

        time_start = get_time();
        relu(fm0, 4 * 4 * 512);
        //tconv(fm0, fm1, tconv1_w, tconv1_b, 4, 4, 512, 256);
        time_end = get_time();
        result_time[2] += time_end - time_start;

        //feature map 0
        time_start = get_time();
        w_in = 4; h_in = 4; C = 512; K = 256;
        err = clEnqueueWriteBuffer(queue, bfm0, CL_FALSE, 0, sizeof(float) * w_in * h_in * C, fm0, 0, NULL, NULL);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bfm0);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bfm1);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &btconv1_w);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &btconv1_b);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 4, sizeof(cl_int), &h_in);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 5, sizeof(cl_int), &w_in);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 6, sizeof(cl_int), &C);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 7, sizeof(cl_int), &K);
        CHECK_ERROR(err);

        global_size[2] = K; global_size[1] = 2 * h_in; global_size[0] = 2 * w_in;
        local_size[2] = 16; local_size[1] = 4; local_size[0] = 4;

        clEnqueueNDRangeKernel(
                queue,
                kernel,
                3,
                NULL,
                global_size,
                local_size,
                0,
                NULL,
                NULL);

        err = clEnqueueReadBuffer(queue, bfm1, CL_TRUE, 0, sizeof(float) * 4 * w_in * h_in * K, fm1, 0, NULL, NULL);
        CHECK_ERROR(err);
        time_end = get_time();
        result_time[3] += time_end - time_start;

        time_start = get_time();
        batch_norm(fm1, bn1_beta, bn1_gamma, bn1_mean, bn1_var, 8 * 8, 256);
        time_end = get_time();
        result_time[4] += time_end - time_start;

        time_start = get_time();
        relu(fm1, 8 * 8 * 256);
        //tconv(fm1, fm2, tconv2_w, tconv2_b, 8, 8, 256, 128);
        time_end = get_time();
        result_time[5] += time_end - time_start;

        //feature map 1
        time_start = get_time();
        w_in *= 2; h_in *= 2; C /= 2; K /= 2;
        err = clEnqueueWriteBuffer(queue, bfm1, CL_FALSE, 0, sizeof(float) * w_in * h_in * C, fm1, 0, NULL, NULL);
        CHECK_ERROR(err);

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bfm1);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bfm2);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &btconv2_w);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &btconv2_b);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 4, sizeof(cl_int), &h_in);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 5, sizeof(cl_int), &w_in);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 6, sizeof(cl_int), &C);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 7, sizeof(cl_int), &K);
        CHECK_ERROR(err);

        global_size[2] = K; global_size[1] = 2 * h_in; global_size[0] = 2 * w_in;
        local_size[2] = 16; local_size[1] = 4; local_size[0] = 4;

        clEnqueueNDRangeKernel(
                queue,
                kernel,
                3,
                NULL,
                global_size,
                local_size,
                0,
                NULL,
                NULL);

        err = clEnqueueReadBuffer(queue, bfm2, CL_TRUE, 0, sizeof(float) * 4 * w_in * h_in * K, fm2, 0, NULL, NULL);
        CHECK_ERROR(err);
        time_end = get_time();
        result_time[6] += time_end - time_start;

        time_start = get_time();
        batch_norm(fm2, bn2_beta, bn2_gamma, bn2_mean, bn2_var, 16 * 16, 128);
        time_end = get_time();
        result_time[7] += time_end - time_start;

        time_start = get_time();
        relu(fm2, 16 * 16 * 128);
        time_end = get_time();
        result_time[8] += time_end - time_start;
        //tconv(fm2, fm3, tconv3_w, tconv3_b, 16, 16, 128, 64);


        //feature map 2
        time_start = get_time();
        w_in *= 2; h_in *= 2; C /= 2; K /= 2;
        err = clEnqueueWriteBuffer(queue, bfm2, CL_FALSE, 0, sizeof(float) * w_in * h_in * C, fm2, 0, NULL, NULL);
        CHECK_ERROR(err);


        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bfm2);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bfm3);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &btconv3_w);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &btconv3_b);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 4, sizeof(cl_int), &h_in);   
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 5, sizeof(cl_int), &w_in);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 6, sizeof(cl_int), &C);   
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 7, sizeof(cl_int), &K);
        CHECK_ERROR(err);

        global_size[2] = K; global_size[1] = 2 * h_in; global_size[0] = 2 * w_in;
        local_size[2] = 16; local_size[1] = 4; local_size[0] = 4;

        clEnqueueNDRangeKernel(
                queue,
                kernel,
                3,
                NULL,
                global_size,
                local_size,
                0,
                NULL,
                NULL);

        err = clEnqueueReadBuffer(queue, bfm3, CL_TRUE, 0, sizeof(float) * 4 * w_in * h_in * K, fm3, 0, NULL, NULL);
        CHECK_ERROR(err);
        time_end = get_time();
        result_time[9] += time_end - time_start;

        time_start = get_time();
        batch_norm(fm3, bn3_beta, bn3_gamma, bn3_mean, bn3_var, 32 * 32, 64);
        time_end = get_time();
        result_time[10] += time_end - time_start;

        time_start = get_time();
        relu(fm3, 32 * 32 * 64);

        time_end = get_time();
        result_time[11] += time_end - time_start;
        //tconv(fm3, output, tconv4_w, tconv4_b, 32, 32, 64, 3);

        //feature map 3
        time_start = get_time();
        w_in *= 2; h_in *= 2; C /= 2; K = 3;
        err = clEnqueueWriteBuffer(queue, bfm3, CL_FALSE, 0, sizeof(float) * w_in * h_in * C, fm3, 0, NULL, NULL);
        CHECK_ERROR(err);

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bfm3);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &boutput);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &btconv4_w);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &btconv4_b);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 4, sizeof(cl_int), &h_in);   
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 5, sizeof(cl_int), &w_in);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 6, sizeof(cl_int), &C);   
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel, 7, sizeof(cl_int), &K);
        CHECK_ERROR(err);

        global_size[2] = K; global_size[1] = 2 * h_in; global_size[0] = 2 * w_in;
        local_size[2] = 1; local_size[1] = 16; local_size[0] = 16;

        clEnqueueNDRangeKernel(
                queue,
                kernel,
                3,
                NULL,
                global_size,
                local_size,
                0,
                NULL,
                NULL);

        err = clEnqueueReadBuffer(queue, boutput, CL_TRUE, 0, sizeof(float) * 4 * w_in * h_in * K, output, 0, NULL, NULL);
        CHECK_ERROR(err);
        time_end = get_time();
        result_time[12] += time_end - time_start;

        time_start = get_time();
        tanh_layer(output, 64 * 64 * 3);
        time_end = get_time();
        result_time[13] += time_end - time_start;
    }

    for(loop = 0; loop < 5; loop++){
        printf("\nnumber%d : %0.9f \n",loop, result_time[loop]);
        printf("\nnumber%d : %0.9f \n",loop, result_time[loop+1]);
        if(loop != 4){
            printf("\nnumber%d : %0.9f \n", loop, result_time[loop+2]);
        }
    }
    // free resources
    free(fm0);
    free(fm1);
    free(fm2);
    free(fm3);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

}
