__kernel void tconv(__global float* in, __global float* out, __global float* weight __global float* bias, int H_IN, int W_IN, int C, int K)
{
        int w_out = get_global_id(2);
        int h_out = get_global_id(1);
        int k = get_global_id(0);
        float sum = 0.0f;
        int H_OUT = H_IN * 2, W_OUT = W_IN * 2;
        int r, s, h_in, w_in, c;

        for(r = 0; r < 5; r++)
        {
            for(s = 0; s < 5; s++)
            {
                // Top & left padding = 3, bottom & rigt padding = 2
                h_in = h_out - 3 + r;
                w_in = w_out - 3 + r;

                if(h_in % 2 == 0 && w_in % 2 == 0)
                {
                    h_in /= 2;
                    w_in /= 2;

                    // Boundary Check
                    if(0 <= h_in && h_in < H_IN && 0 <= w_in && w_in < W_IN)
                    {
                        for(c = 0; c < C; c++)
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
