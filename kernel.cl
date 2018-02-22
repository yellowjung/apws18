__kernel void tconv_k(__global float* in, __global float* out, __global float* weight, __global float* bias, int H_IN, int W_IN, int C, int K)
{
    int w_out = get_global_id(0);
    int h_out = get_global_id(1);
    int k = get_global_id(2);
    float sum = 0.0f;
    int H_OUT = H_IN * 2, W_OUT = W_IN * 2;
    int r, s, h_in, w_in, c;

    for(r = 0; r < 5; ++r)
    {
      for(s = 0; s < 5; ++s)
      {
        // Top & left padding = 3, bottom & rigt padding = 2
        h_in = h_out - 3 + r;
        w_in = w_out - 3 + s;

        if(h_in % 2 == 0 && w_in % 2 == 0)
        {
          h_in /= 2;
          w_in /= 2;

          // Boundary Check
          if(0 <= h_in && h_in < H_IN && 0 <= w_in && w_in < W_IN)
          {
            for(c = 0; c < C; ++c)
            {
              // Filter is stored in reverse. use [4 - r][4 - s]
              // sum += in[h_in][w_in][c] * weight[4 - r][4 - s][k][c]
              sum += in[(h_in * W_IN + w_in) * C + c] * weight[(((4 - r) * 5 + (4 - s)) * K + k) * C + c];
            }
          }
        }
      }
    }

    sum += bias[k];
    // out[h_out][w_out][k] = sum
    out[(h_out * W_OUT + w_out) * K + k] = sum;
}

__kernel void batch_norm_k(__global float* inout, __global float* beta, __global float* gamma, __global float* mean, __global float* var, int HW, int C)
{
    int i = get_global_id(0);
    int c = get_global_id(1);
    float scaled_gamma;
    scaled_gamma = gamma[c] / (float)sqrt(var[c] + 1e-5);
    inout[i * C + c] = scaled_gamma * inout[i * C + c] + (beta[c] - scaled_gamma * mean[c]);
}

__kernel void relu_k(__global float* inout, int HWC)
{
    int i = get_global_id(0);

    inout[i] = fmax(inout[i], 0);
}

__kernel void tanh_layer_k(__global float* inout, int HWC)
{
    int i = get_global_id(0);

    inout[i] = tanh(inout[i]);
}

__kernel void proj_k(__global float* in, __global float* out, __global float* weight, __global float* bias, int C, int K)
{
    int i = get_global_id(1);
    int l_i = get_local_id(1);
    __local float sum[256];
    int c = get_local_id(0);
    
    sum[l_i * 128 + c] = (c < C) ? in[c] * weight[c * K + i] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int p = get_local_size(0) / 2; p >= 1; p = p >> 1)
    {
        if(c < p)
            sum[l_i * 128 + c] += sum[l_i * 128 + c + p];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(c == 0)
    {
        out[i] = sum[l_i * 128] + bias[i];
    }
/*
    int i = get_global_id(0);
    float sum = 0.0f;
    int c;

    for(c = 0; c < C; ++c)
    {
        sum += in[c] * weight[c * K + i];
    }

    sum += bias[i];

    out[i] = sum;
*/
}

