# Compte rendu - PPAR GPU Lab 0 & 1

### Brandon Largeau - Thomas Esseul

## Question 1

```C
float log2_series_brandon(int n)
{
    float result = 0.0F;

    for (int i = 0; i < n; ++i)
    {
        result += (powf(-1, i)) / (i + 1);
    }

    return result;
}

float log2_series_thomas(int n)
{
    float res = 0.0F;

    int op = 1;

    for(int i=1; i<=n; i++)
    {
        res += (float) 1/i * op;
        op *= -1;
    }

    return res;
}
```

## Question 2

On contaste une variation de résultats quand on calcule de 1 à N et de N à 1. Cette différence peut s'expliquer par la non-assossiativité de la soustraction. Pour se rendre compte de la différence de résultat, il faut afficher un plus grand nombre de chiffres après la virgule (`%.17f`).

## Question 3

On utilisera l'implémentation ``float log2_series_thomas(int n)`` pour l'élaboration de nos différentes solutions (voir plus haut pour l'implémentation).

### Solution 1

Pour la première solution, le thread ``i`` calculera les résultats pour l'intervalle `[i * (n/m); (i + 1) * (n/m)[`, en fera la somme et la retourna au CPU.

Exemple avec ``m = 3`` threads et ``n = 9`` : 

- Thread_1 ``[0; 3[``
- Thread_2 ``[3; 6[``
- Thread_3 ``[6; 9[``

### Solution 2

Pour la seconde solution, le thread ``i`` calculera les résultats pour les données avec un pas égale à ``m``.

Exemple avec ``m = 3`` threads, ``n = 9`` et donc un pas de ``m = 3`` :

- Thread_1 ``0, 3, 6``
- Thread_2 ``1, 4, 7``
- Thread_3 ``2, 5, 8``

## Question 4

On a alloué sur le CPU la zone mémoire qui recevra les données calculées par le GPU :

```C
float* data_out_cpu = (float *)calloc(results_size, sizeof(float));
```

Utilisation d'un ``calloc()`` pour directement initialiser la mémoire à ``0``.

On a alloué sur le GPU une zone mémoire.

``float* data_out_gpu;`` -> contiendra le résultat des calculs pour être utilisé en sortie :

```C
cudaMalloc((void **)&data_out_gpu, alloc_size);

cudaMemset((void *)data_out_gpu, 0, alloc_size);
```

Chaque thread doit gérer ``data_size / num_threads`` données.

Il y aura l'exécution du kernel.

On copie le résultat du GPU vers le CPU :

```C
cudaMemcpy(data_out_cpu, data_out_gpu, alloc_size, cudaMemcpyDeviceToHost);
```

Il y aura la réduction.

On libère les ressources allouées :

```C
cudaFree(data_out_gpu);
free(data_out_cpu);
```

## Question 5

On implémente sur GPU la Solution 1, `ind` représente ici l'index global du thread puisque dans cette version on n'utilise pas la notion de bloc.
```C
__global__ void summation_kernel(int data_size, float* data_out)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    float res = 0.0F;
    int op = -1;

    for(int j = ind * data_size; j < (ind + 1) * data_size; j++)
    {
        res += j == 0 ? 0 : (float) 1 / j * op;
        op *= -1;
    }

    data_out[ind] = res;
}
```

Implémentation de la solution 2 avec le ``pas``, égale ici à ``(i * num_threads)``.

```C
__global__ void summation_kernel_2(int data_size, float* data_out)
{
    int threadNumber = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    int op;
    float res = 0.0F;

    for (int i = 0; i < data_size; ++i)
    {
        op = (threadNumber % 2 == 0) ? -1 : 1;

        res += (i == 0 && threadNumber == 0) ? 0 : (float) 1 / (threadNumber + (i * num_threads)) * op;
    }

    data_out[threadNumber] = res;
}
```

La réduction est faite sur le CPU :

```C
float sum = 0.0F;

for (int i = 0; i < results_size; ++i)
{
    sum += data_out_cpu[i];
}
```

## Question 6

Avec une première implémentation, on obtient l'output 

```
CPU result: 0.693138
 log(2)=0.693147
 time=0.641260s
GPU results:
 Sum: 0.693138
 Total time: 0.019142 s,
 Per iteration: 0.142619 ns
 Throughput: 28.0468 GB/s
```

Sans surprise, on note une différence nette entre la version CPU et la version GPU, explicable par l'utilisation de parallélisme pour la version du processeur graphique.

La seconde implémentation retourne : 

```
CPU result: 0.693138
 log(2)=0.693147
 time=0.642594s
GPU results:
 Sum: 0.692439
 Total time: 0.0192812 s,
 Per iteration: 0.143656 ns
 Throughput: 27.8443 GB/s
```

On note qu'avec cette implémentation il y a une différence sur le résultat (peut-être un problème au niveau de l'implémentation?) tout en ayant toujours une meilleure performance que la version CPU.

## Question 7

Sur la machine utilisée pour les tests, la configuration avec l'exécution la plus rapide est de 16 blocs avec 128 threads par bloc, ce qui s'explique notamment par le nombre important de threads utilisés.

## Question 8

(Fait sur ``summation_kernel()``)

On peut réduire la quantité de données retournée au CPU en ne retournant qu'une donnée par bloc. Pour ce faire, le thread ``0`` de chaque bloc fera la somme des valeurs calculées par le bloc.

Les valeurs intermédiaires calculées par chaque thread seront alors stockées en mémoire partagée ``s_res``, la mémoire partagée a ici un montant ``sMem_size`` représentant le ``nombre de threads dans un bloc * sizeof(float)`` défini lors de l'appel de la fonction.

Chaque bloc a sa propre version de cette mémoire partagée, elle est partagée entre les threads d'un même bloc, mais pas entre chaque bloc.

Chaque bloc doit avoir terminé son travail avant de faire le résultat final de ce bloc, c'est pour cela qu'on utilise le principe de barrière avec ``__syncthreads();``.

Les résultats sont stockés dans la mémoire globale ``data_out`` (utile pour la question 9 qui plus est).

```C
// GPU kernel
// data_size = data_size_per_thread
__global__ void summation_kernel(int data_size, float* data_out)
{
    // Question 8
    extern __shared__ float s_res[];

    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float res = 0.0F;
    int op = -1;

    for(int j = ind * data_size; j < (ind + 1) * data_size; j++)
    {
        res += j == 0 ? 0 : (float) 1 / j * op;
        op *= -1;
    }

    //data_out[ind] = res;

    // Question 8

    s_res[tid] = res;

    __syncthreads();

    if(tid == 0)
    {
        for(int i = 1; i < blockDim.x; i++)
        {
            res += s_res[i];
        }

        data_out[blockIdx.x] = res;
    }
}
```

Bien entendu la mémoire allouée pour ``data_out`` est trop élevée, on devrait passer de ``num_threads`` à ``blocks_in_grid``.

On a gardé la même boucle de réduction sur le CPU qui itère sur tous les élements de ``data_out_cpu`` (même si on utilise que ``blocks_in_grid`` cases). Le résultat est le même étant donné que les cases non utilisées sont à ``0``. On sait que ce n'est pas ce qui est le mieux, mais de cette façon on garde un code CPU qui fonctionne pour toutes les implémentations, pour les besoins du TP.

## Question 9

(Fait sur ``summation_kernel()``)

Ici c'est le thread ``0`` du bloc ``0`` qui fera le calcul final.

On attend le résultat final de chaque bloc avant de faire le calcul final, utilisation de ``__syncthreads();``.

On met ``res`` à ``0.0F`` car on va l'utiliser plus bas.

On accumule les résultats de chaque bloc dans la variable ``res``, on met les ``gridDim.x`` premiers éléments de ``data_out`` à ``0`` (c'est les seuls cases utilisées, le reste est à ``0``) et on stocke le résultat à l'indice ``0`` de ``data_out``.

```C
// GPU kernel
// data_size = data_size_per_thread
__global__ void summation_kernel(int data_size, float* data_out)
{
    // Question 8
    extern __shared__ float s_res[];

    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float res = 0.0F;
    int op = -1;

    for(int j = ind * data_size; j < (ind + 1) * data_size; j++)
    {
        res += j == 0 ? 0 : (float) 1 / j * op;
        op *= -1;
    }

    //data_out[ind] = res;

    // Question 8

    s_res[tid] = res;

    __syncthreads();

    if(tid == 0)
    {
        for(int i = 1; i < blockDim.x; i++)
        {
            res += s_res[i];
        }

        data_out[blockIdx.x] = res;
    }

    // Question 9

    __syncthreads();

    res = 0.0F;

    if (ind == 0)
    {
        for (int i = 0; i < gridDim.x; ++i)
        {
            res += data_out[i];
        }

        // Clean memory of the first "gridDim.x" elements of the global memory "data_out"
        // because this is the only things being modified, the rest are only 0
        memset(data_out, 0, gridDim.x);

        // store the final result in the first indice (0)
        data_out[0] = res;
    }
}
```

Bien entendu la mémoire allouée pour ``data_out`` est trop élevée, on devrait passer de ``num_threads`` à ``blocks_in_grid``.

On a gardé la même boucle de réduction sur le CPU qui itère sur tous les élements de ``data_out_cpu`` (même si on utilise que ``1`` case). Le résultat est le même étant donné que les cases non utilisées sont à ``0``. On sait que ce n'est pas ce qui est le mieux, mais de cette façon on garde un code CPU qui fonctionne pour toutes les implémentations, pour les besoins du TP.
