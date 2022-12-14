/* Université Paris 8 - 2022 */
/* Superviseur : Pr Youcef TOUATI */
/* -- */
/* Author 1 : ADJIBADE Ahmed */
/* Author 2 : SAMMATRICE Lorenzo */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>

#include "models.h"

#define SIZEOF(arr) sizeof(arr) / sizeof(*arr)
double LEARNING_RATE = 0.01; // default value 0.01

/* OPERATORS VALUES : AND, OR, XOR */
entries *AND_ENTRIES[4];
outputs *AND_OUTPUTS[4];
entries *OR_ENTRIES[4];
outputs *OR_OUTPUTS[4];
entries *XOR_ENTRIES[4];
outputs *XOR_OUTPUTS[4];

double random_double() { return ((float)rand()) / ((float)RAND_MAX); }

void init_operators_values()
{
    // allocation of memory
    for (int i = 0; i < 4; i++)
    {
        AND_ENTRIES[i] = malloc(sizeof(entries));
        AND_OUTPUTS[i] = malloc(sizeof(outputs));
        OR_ENTRIES[i] = malloc(sizeof(entries));
        OR_OUTPUTS[i] = malloc(sizeof(outputs));
        XOR_ENTRIES[i] = malloc(sizeof(entries));
        XOR_OUTPUTS[i] = malloc(sizeof(outputs));
    }
    //  AND  values
    AND_ENTRIES[0]->x1 = 0;
    AND_ENTRIES[0]->x2 = 0;
    AND_OUTPUTS[0]->y = 0;

    AND_ENTRIES[1]->x1 = 0;
    AND_ENTRIES[1]->x2 = 1;
    AND_OUTPUTS[1]->y = 0;

    AND_ENTRIES[2]->x1 = 1;
    AND_ENTRIES[2]->x2 = 0;
    AND_OUTPUTS[2]->y = 0;

    AND_ENTRIES[3]->x1 = 1;
    AND_ENTRIES[3]->x2 = 1;
    AND_OUTPUTS[3]->y = 1;

    //  OR  values
    OR_ENTRIES[0]->x1 = 0;
    OR_ENTRIES[0]->x2 = 0;
    OR_OUTPUTS[0]->y = 0;

    OR_ENTRIES[1]->x1 = 0;
    OR_ENTRIES[1]->x2 = 1;
    OR_OUTPUTS[1]->y = 1;

    OR_ENTRIES[2]->x1 = 1;
    OR_ENTRIES[2]->x2 = 0;
    OR_OUTPUTS[2]->y = 1;

    OR_ENTRIES[3]->x1 = 1;
    OR_ENTRIES[3]->x2 = 1;
    OR_OUTPUTS[3]->y = 1;

    // XOR  values
    XOR_ENTRIES[0]->x1 = 0;
    XOR_ENTRIES[0]->x2 = 0;
    XOR_OUTPUTS[0]->y = 0;

    XOR_ENTRIES[1]->x1 = 0;
    XOR_ENTRIES[1]->x2 = 1;
    XOR_OUTPUTS[1]->y = 1;

    XOR_ENTRIES[2]->x1 = 1;
    XOR_ENTRIES[2]->x2 = 0;
    XOR_OUTPUTS[2]->y = 1;

    XOR_ENTRIES[3]->x1 = 1;
    XOR_ENTRIES[3]->x2 = 1;
    XOR_OUTPUTS[3]->y = 0;
}

void set_inputs(int x1, int x2, rna *RNA)
{
    RNA->INPUT[0] = x1;
    RNA->INPUT[1] = x2;
}

void init_rna(rna *RNA)
{
    /* weights */
    for (int i = 0; i < NBR_INPUT; i++)
    {
        for (int j = 0; j < NBR_NEURON_HIDDEN_LAYER; j++)
        {
            RNA->WEIGHT1[i][j] = random_double();
        }
    }
    for (int i = 0; i < NBR_NEURON_HIDDEN_LAYER; i++)
    {
        for (int j = 0; j < NBR_OUTPUT; j++)
        {
            RNA->WEIGHT2[i][j] = random_double();
        }
    }

    /* biases */
    for (int i = 0; i < NBR_INPUT; i++)
    {
        for (int j = 0; j < NBR_NEURON_HIDDEN_LAYER; j++)
        {
            RNA->BIAS1[i][j] = random_double();
        }
    }
    for (int i = 0; i < NBR_NEURON_HIDDEN_LAYER; i++)
    {
        for (int j = 0; j < NBR_OUTPUT; j++)
        {
            RNA->BIAS2[i][j] = random_double();
        }
    }

    /* init neurons */
    for (int i = 0; i < NBR_HIDDEN_LAYER; i++)
    {
        for (int j = 0; j < NBR_NEURON_HIDDEN_LAYER; j++)
        {
            RNA->NEURONS[i][j] = malloc(sizeof(neuron));
        }
    }
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x)
{
    return x * (1 - x);
}

int propagation(rna *RNA) // return output value
{
    /* from entry to first hidden layer */
    for (int i = 0; i < NBR_NEURON_HIDDEN_LAYER; i++)
    {
        double x = 0;
        for (int j = 0; j < NBR_INPUT; j++)
        {
            x += RNA->INPUT[j] * RNA->WEIGHT1[j][i] + RNA->BIAS1[j][i];
        }
        RNA->NEURONS[0][i]->output = sigmoid(x);
    }

    /* from first hidden layer to output - we have only one output*/
    double x = 0;
    for (int j = 0; j < NBR_NEURON_HIDDEN_LAYER; j++)
    {
        x += RNA->NEURONS[0][j]->output * RNA->WEIGHT2[j][0] + RNA->BIAS2[j][0];
    }
    RNA->output = sigmoid(x);
    if (RNA->output >= 0.5)
    {
        return 1;
    }
    else
    {
        return 0;
    }

    // return output;
}

void back_propagation(int expected_output, rna *RNA)
{
    double error = expected_output - RNA->output;
    double delta = error * sigmoid_derivative(RNA->output);
    for (int i = 0; i < NBR_NEURON_HIDDEN_LAYER; i++)
    {
        double weight_delta = delta * RNA->NEURONS[0][i]->output;
        RNA->WEIGHT2[i][0] += LEARNING_RATE * weight_delta;
        RNA->BIAS2[i][0] += LEARNING_RATE * delta;
    }

    for (int i = 0; i < NBR_NEURON_HIDDEN_LAYER; i++)
    {
        double error = delta * RNA->WEIGHT2[i][0];
        double delta = error * sigmoid_derivative(RNA->NEURONS[0][i]->output);
        for (int j = 0; j < NBR_INPUT; j++)
        {
            double weight_delta = delta * RNA->INPUT[j];
            RNA->WEIGHT1[j][i] += LEARNING_RATE * weight_delta;
            RNA->BIAS1[j][i] += LEARNING_RATE * delta;
        }
    }
}

/* universal learning function */

void learn(rna *RNA, char *name, entries *ENTRY_VECTOR[], outputs *OUTPUT_VECTOR[], boolean verbose)
{
    verbose == T ? printf("\n") : 0;
    for (int i = 0; i < 4; i++)
    {
        set_inputs(ENTRY_VECTOR[i]->x1, ENTRY_VECTOR[i]->x2, RNA);
        int output = 0;
        int counter = 0;
        do
        {
            counter++;
            output = propagation(RNA);
            if (output != OUTPUT_VECTOR[i]->y)
            {
                back_propagation(OUTPUT_VECTOR[i]->y, RNA);
            }
            if (counter > MAX_LEARNING_ITERATION)
            {
                verbose == T ? printf("Error: max learning times reached, on this example\n") : 0;
                break;
            }

        } while (output != OUTPUT_VECTOR[i]->y);
        verbose == T ? printf("%s Example %d: %d %d -> %d (expected %d) after %d learning times \n", name, i, ENTRY_VECTOR[i]->x1, ENTRY_VECTOR[i]->x2, output, OUTPUT_VECTOR[i]->y, counter) : 0;
    }
    verbose == T ? printf("\n") : 0;
}

void train(char *name, rna *RNA, entries *ENTRY_VECTOR[], outputs *OUTPUT_VECTOR[], boolean verbose)
{
    init_rna(RNA);
    learn(RNA, name, ENTRY_VECTOR, OUTPUT_VECTOR, verbose);
    printf("--------------------\n");
}

/* utils functions */
void weights_printer(rna *RNA)
{
    printf("WEIGHT1 : \n");
    for (int i = 0; i < NBR_INPUT; i++)
    {
        for (int j = 0; j < NBR_NEURON_HIDDEN_LAYER; j++)
        {
            printf("%f ", RNA->WEIGHT1[i][j]);
        }
        printf("\n");
    }
    printf("WEIGHT2 : \n");
    for (int i = 0; i < NBR_NEURON_HIDDEN_LAYER; i++)
    {
        for (int j = 0; j < NBR_OUTPUT; j++)
        {
            printf("%f ", RNA->WEIGHT2[i][j]);
        }
        printf("\n");
    }
}

void biases_printer(rna *RNA)
{
    printf("BIAS1 : \n");
    for (int i = 0; i < NBR_INPUT; i++)
    {
        for (int j = 0; j < NBR_NEURON_HIDDEN_LAYER; j++)
        {
            printf("%f ", RNA->BIAS1[i][j]);
        }
        printf("\n");
    }
    printf("BIAS2 : \n");
    for (int i = 0; i < NBR_NEURON_HIDDEN_LAYER; i++)
    {
        for (int j = 0; j < NBR_OUTPUT; j++)
        {
            printf("%f ", RNA->BIAS2[i][j]);
        }
        printf("\n");
    }
}

void free_memory(rna *RNA[]) // to avoid memory leaks
{
    for (int z = 0; z < 2; z++)
    {
        for (int i = 0; i < NBR_HIDDEN_LAYER; i++)
        {
            for (int j = 0; j < NBR_NEURON_HIDDEN_LAYER; j++)
            {
                free(RNA[z]->NEURONS[i][j]);
            }
        }
    }

    for (int i = 0; i < 4; i++)
    {
        free(AND_ENTRIES[i]);
        free(AND_OUTPUTS[i]);
        free(XOR_ENTRIES[i]);
        free(XOR_OUTPUTS[i]);
    }
}

/* end of utils functions */

/*  threads functions */
void *thread_and_handler(void *arg)
{
    train("AND", arg, AND_ENTRIES, AND_OUTPUTS, T);
    pthread_exit(NULL);
}

void *thread_xor_handler(void *arg)
{
    train("OR", arg, OR_ENTRIES, OR_OUTPUTS, T);
    pthread_exit(NULL);
}

pthread_mutex_t rna_mutex_thread = PTHREAD_MUTEX_INITIALIZER;

/* main program */
int main()
{
    // init  :: do not remove thoses lines or you will have a segmentation fault !!
    init_operators_values();

    // RNAs
    rna *RNA_AND = malloc(sizeof(rna));
    rna *RNA_OR = malloc(sizeof(rna));

    // we will use threads to train the RNAs
    pthread_t thread1, thread2;
    pthread_create(&thread1, NULL, thread_and_handler, RNA_AND);
    pthread_create(&thread2, NULL, thread_xor_handler, RNA_OR);

    pthread_mutex_lock(&rna_mutex_thread);
    pthread_join(thread1, NULL);
    pthread_mutex_unlock(&rna_mutex_thread);

    pthread_join(thread2, NULL);

    /* tests */
    printf("--------------------\nTesting ... \n");
    int x1 = 0, x2 = 0;
    set_inputs(x1, x2, RNA_AND);
    set_inputs(x1, x2, RNA_OR);
    printf("Entries %d, %d\n", x1, x2);
    printf("AND output :  %d\n", propagation(RNA_AND));
    printf("OR output :  %d\n", propagation(RNA_OR));
    printf("End of testing\n --------------------\n");

    /* free memory */
    rna *RNAS[] = {RNA_AND, RNA_OR};
    free_memory(RNAS);
    return 0;
}