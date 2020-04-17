---
tags: M1-PPAR
---

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

On utilisera l'implémentation ``float log2_series_brandon(int n)`` pour l'élaboration de nos différentes solutions (voir plus haut pour l'implémentation).

### Solution 1

Pour la première solution, le thread ``i`` calculera les résultats pour l'intervalle `[i * (n/m); (i + 1) * (n/m)[`, en fera la somme et la retourna au CPU.

Exemple avec ``m = 3`` threads et ``n = 9`` : 

- Thread_1 ``[0; 3[``
- Thread_2 ``[3; 6[``
- Thread_3 ``[6; 9[``

### Solution 2

Pour la seconde solution, le thread ``i`` calculera les résultats pour les données avec un pas égale à ``n/m``.

Exemple avec ``m = 3`` threads, ``n = 9`` et donc un pas de ``n/m = 3`` :

- Thread_1 ``0, 3, 6``
- Thread_2 ``1, 4, 7``
- Thread_3 ``2, 5, 8``

## Question 4

On a alloué sur le GPU 2 zones mémoire.

``float* data_out_gpu;`` -> contiendra le résultat des calucls pour être utilisé en sortie

``int* data_size_gpu;`` -> contiendra ``data_size`` permettant son utilisation dans le code du GPU

```C
cudaMalloc((void **)&data_out_gpu, alloc_size);
cudaMalloc((void **)&data_size_gpu, sizeof(int));
```

On copie ensuite ``data_size`` dans ``data_size_gpu`` :

```C
cudaMemcpy(data_size_gpu, data_size, sizeof(int), cudaMemcpyHostToDevice);
```

Il y aura l'exécution du kernel.

On copie le résultat du GPU vers le CPU :

```C
cudaMemcpy(data_out_cpu, data_out_gpu, alloc_size, cudaMemcpyDeviceToHost);
```

Il y aura la réduction.

On libère les ressources allouées :

```C
cudaFree(data_size_gpu);
cudaFree(data_out_gpu);
free(data_out_cpu);
```

## Question 5

...
