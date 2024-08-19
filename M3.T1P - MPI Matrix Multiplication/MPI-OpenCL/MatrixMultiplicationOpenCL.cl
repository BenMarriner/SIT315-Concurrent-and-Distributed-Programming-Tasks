kernel void multiply(__global int* a, __global int* b, __global int* c, const int M, const int N, const int K)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    int value = 0;
    for (int k = 0; k < N; k++)
    {
        value += a[i * N + k] * b[k * N + j];
    }

    c[i * N + j] = value;
}