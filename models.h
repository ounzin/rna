/* RNA VALUES */
#define NBR_INPUT 2
#define NBR_OUTPUT 1
#define NBR_HIDDEN_LAYER 1
#define NBR_NEURON_HIDDEN_LAYER 4
#define MAX_LEARNING_ITERATION 10000

/* structures descriptions */

/* boolean */
typedef enum
{
   F,
   T
} boolean;

/* neuron struct */
typedef struct
{
   double delta;
   double output;
} neuron;

/* rna struct */

typedef struct
{
   double INPUT[NBR_INPUT];
   double WEIGHT1[NBR_INPUT][NBR_NEURON_HIDDEN_LAYER];
   double WEIGHT2[NBR_NEURON_HIDDEN_LAYER][NBR_OUTPUT];
   double BIAS1[NBR_INPUT][NBR_NEURON_HIDDEN_LAYER];
   double BIAS2[NBR_NEURON_HIDDEN_LAYER][NBR_OUTPUT];
   double LEARNING_RATE;
   double output;
   neuron *NEURONS[NBR_HIDDEN_LAYER][NBR_NEURON_HIDDEN_LAYER];
} rna;

/* entries */
typedef struct
{
   int x1;
   int x2;
} entries;

/* output */
typedef struct
{
   int y;
} outputs;