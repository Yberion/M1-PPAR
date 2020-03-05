
/* Sequential algorithm for converting a text in a digit sequence
 *
 * PPAR, TP3
 *
 * A. Mucherino
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#pragma warning(disable : 4996) // Disable MSVC warning 4996: This function or variable may be unsafe. Consider using safe-version instead.

int main(int argc, char* argv[])
{
    size_t i;
    long fileSize;
    int count;
    unsigned char* text;
    char fileName[20] = { 0 };
    int charact;
    bool firstInteger = true;
    bool canPutComma = false;
    FILE* filePtr;

    if (argc > 1)
    {
        // getting started (we suppose that the argv[1] contains the filename related to the text)
        strcpy(fileName, argv[1]);
        filePtr = fopen(fileName, "r");
    }
    else
    {
        strcpy(fileName, "pi-text.txt");
        filePtr = fopen(fileName, "r");
    }

    if (!filePtr)
    {
        fprintf(stderr, "%s (%s): unable to open file '%s', stopping\n", argv[0], __func__, fileName);
        return EXIT_FAILURE;
    }

    // put the cursor at the end of the file
    fseek(filePtr, 0L, SEEK_END);

    // get the file size
    fileSize = ftell(filePtr);

    if (fileSize == -1)
    {
        fprintf(stderr, "%s (%s): error reading file size on file '%s', stopping\n", argv[0], __func__, fileName);
        return EXIT_FAILURE;
    }

    // put the cursor at the beginning of the file
    rewind(filePtr);

    text = (unsigned char *)calloc((size_t)fileSize + 1, sizeof(unsigned char));

    if (text == NULL)
    {
        fprintf(stderr, "%s (%s): unable to allocate memory to contain the text of file '%s', stopping\n", argv[0], __func__, fileName);
        return EXIT_FAILURE;
    }

    i = 0;

    // reading the text
    while ((charact = fgetc(filePtr)) != EOF)
    {
        text[i] = (char)charact;
        ++i;
    }

    count = 0;

    // Works fine, checked on http://www.geom.uiuc.edu/~huberty/math5337/groupe/digits.html
    // converting the text
    for (i = 0; i < (size_t)fileSize; ++i)
    {
        if (firstInteger && canPutComma)
        {
            printf(",");

            firstInteger = false;
        }

        // 0-9
        if (text[i] >= 48 && text[i] <= 57)
        {
            if (count)
            {
                printf("%d", count);

                count = 0;
            }

            printf("%c", text[i]);

            canPutComma = true;
            continue;
        }

        // blank, point, new line
        if (text[i] == 32 || text[i] == 46 || text[i] == 10)
        {
            if (count)
            {
                printf("%d", count);

                count = 0;
            }

            canPutComma = true;
            continue;
        }

        // other special char
        if (text[i] >= 33 && text[i] <= 45 || text[i] == 47 || text[i] >= 58 && text[i] <= 64 || text[i] >= 91 && text[i] <= 96 || text[i] >= 123 && text[i] <= 126)
        {
            if (count)
            {
                printf("%d", count);

                count = 0;
            }

            printf("0");

            canPutComma = true;
            continue;
        }

        // A-Z, a-z
        if ((text[i] >= 65 && text[i] <= 90) || (text[i] >= 97 && text[i] <= 122))
        {
            ++count;
            continue;
        }
    }

    // free calloc
    free(text);
    // close file
    fclose(filePtr);

    // ending
    return EXIT_SUCCESS;
}