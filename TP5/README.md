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
