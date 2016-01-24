#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h> 
#include <time.h>

/* Last Modified  November 28, 2011, 10:06am  */

/********************  Begining of the Header ********************/ 

#define INPUTS       "MgT_Varied.dat"
#define TARGETS      ""
#define RESULTS      "O-Mg_FF_Results.csv"

#define NF_VAR       3         /* Noise free variable index          */
                               /* NF_VAR = 1,...,INODES              */

#define CORENPAT     102        /* Core number of training patterns   */ 
                               /* in the input file                  */
#define REPLICATIONS 2000        /* Number of replications             */
#define NOISE_LEVEL  0.01      /* Noise level as % of                */
#define NPAT         (CORENPAT*REPLICATIONS)     
                                /* Number of training patterns       */
#define INODES       4          /* Number of input nodes             */
#define ONODES       1          /* Number of nodes in output layer   */
#define HNODES       11         /* Number of nodes in hidden layer   */
#define ACT_FUNC     0          /* Sigmoidal func=0, Arctan func=1   */
                                /* For hidden layer func=1 always    */
#define NORMAL_WS    1          /* Set 1, for normalization          */
                                /* within (intra) signals            */

#define PRINT_TARGET 0          /* Set 1, to print target patterns   */
#define M_MEAN       0.000324   /* Mean of model errors               */
#define M_STDDEV     0.0396     /* SDV of model errors                */
double MinLimit[INODES]={2.5, 50.0000, 206.0000, 0.013032}; 
double MaxLimit[INODES]= {158.70, 300.0000, 420.0000, 0.14951};

/* The following parameters need not be changed                     */
#define NORMAL_AS    0         /* Set 1, for normalization          */
                               /* across (inter) signalrs           */
#define SLOPE1       1.0       /* Slope output node activation func.*/
#define SLOPE2       1.0       /* Slope hidden node activation func.*/
#define SATURATION   100.0     /* Saturation limit of net input     */

#define FLOAT            1
#define DOUBLE           2

void Read_Input_Patterns();
void Normalize_Inputs_Patterns_Across();
void Normalize_Inputs_Patterns_Within();
void Read_Target_Patterns();
void Read_Previous_Iteration(); 
void Normalize_Target_Patterns();
void Write_Bpt2Bpr();
void Read_Network_Weights();
void Read_Node_Thresholds(); 
void Initialize_Network_Weights();
void Initialize_Node_Thresholds(); 
void Initialize_Network_Lag_Weights();
void Initialize_Node_Lag_Thresholds(); 
void Initialize_Network_Activations();
void Initialize_Backpropagation_Error();
FILE *Open_Error_File_To_Write();
FILE *Open_Weight_File_To_Write(); 
FILE *Open_Estimation_File_To_Write();
void Compute_First_Hidden_Layer_Activation();
void Compute_Hidden_Layer_Activation();
void Compute_Second_Hidden_Layer_Activation();
void Compute_Output_Layer_Activation();
void Denormalize_Output_Layer_Activation(); 
void Compute_Error_At_Ouput_Layer();
void Compute_Output_Layer_Error_Gradient();
void Compute_Second_Hidden_Layer_Error_Gradient();
void Compute_First_Hidden_Layer_Error_Gradient();
void Compute_Hidden_Layer_Error_Gradient();
void Modify_Input_Hidden_Layer_Weights();
void Modify_Hidden_Hidden_Layer_Weights();
void Modify_Hidden_Output_Layer_Weights();
void Modify_Hidden_Output_Layer_Weights();
void Modify_First_Hidden_Layer_Thresholds();
void Modify_Hidden_Layer_Thresholds();
void Modify_Second_Hidden_Layer_Thresholds();
void Modify_Output_Layer_Thresholds();
void Write_Asci_Estimates();
void Format_Asci_Estimates();
void Write_Error();
void Write_Weight();
void Write_Cpu_Time();
void Write_Network_Weights();
void Write_Node_Thresholds();
double Gaussian_Random_Number();

char *Vector_Allocate_Char();
float *Vector_Allocate_Float();
double *Vector_Allocate_Double();
int *Vector_Allocate_Int();
float **Matrix_Allocate_Float();
double **Matrix_Allocate_Double();
int **Matrix_Allocate_Int();
double ***Mat_Array_Allocate_Double();
char *Get_String();
int Get_Int();
float Get_Float();
void Error_Msg();

/*
ASSIGN macro:

ASSIGNS VECTOR (a) TO VECTOR (b) (PROMOTING AS REQUIRED).

ASSIGN(a,b,len,typea,typeb)

    a       pointer to first vector (source vector).
    b       pointer to second vector (target vector).
    len     length of vectors (integer).
    typea   legal C type describing the type of a data.
    typeb   legal C type describing the type of b data.
*/
#define ASSIGN(a,b,len,typea,typeb) {  \
                       typea *_PTA = (typea *)a;  \
                       typeb *_PTB = (typeb *)b;  \
                       int _IX;  \
                       for(_IX = 0 ; _IX < (len) ; _IX++)  \
                           *(_PTB)++ = (typeb) (*(_PTA)++);  \
                    }


/*
ASSIGN_MAT MACRO FOR ASSIGNING ONE MATRIX TO THE OTHER MATRIX:

ASSIGN_MAT(a,b,rowsa,colsa,typea,typeb)

    a       pointer to first MATRIX structure (source matrix).
    b       pointer to second MATRIX structure (target matrix).
    rowsa   number of rows in matrix a
    colsa   number of columns in matrix a
    typea   legal C type describing the type of a
    typeb   legal C type describing the type of b
*/
#define ASSIGN_MAT(a,b,rowsa,colsa,typea,typeb) {  \
                 typea **_AMX = (typea **)a;  \
                 typeb **_BMX = (typeb **)b;  \
                 typea *_PTA;  \
                 typeb *_PTB;  \
                 int _IX,_JX;  \
                 for(_IX = 0 ; _IX < rowsa ; _IX++) {  \
                     _PTB = _BMX[_IX];  \
                     _PTA = _AMX[_IX];  \
                     for(_JX = 0 ; _JX < colsa ; _JX++)  \
                         (*_PTB++) = (*_PTA++);  \
                 }  \
             }


/* MATRIX structures */

typedef struct {
    int element_size;
    unsigned int rows;
    unsigned int cols;
    char **ptr;
	       } MATRIX;

typedef struct {
    int element_size;
    unsigned int rows;
    unsigned int cols;
    float **ptr;
	       } MATRIX_FLOAT;

typedef struct {
    int element_size;
    unsigned int rows;
    unsigned int cols;
    double **ptr;
	       } MATRIX_DOUBLE;

/* DATA INFORMATION STRUCTURE FOR MANIPULATING DATA FILES */
typedef struct {
    unsigned char type;          /* data type 1-3 as defined below */
    unsigned char element_size;  /* size of each element */
    unsigned short int records;  /* number of records */
    unsigned short int rec_len;  /* number of elements in each record*/
    char *name;                  /* pointer to file name */
    FILE *fp;                    /* pointer to FILE structure */
} DATA_FILE;

/* FILE HEADER STRUCTURE FOR DATA FILES */
typedef struct {
    unsigned char type;          /* data type 1-3 as defined below */
    unsigned char element_size;  /* size of each element */
    unsigned short int records;  /* number of records */
    unsigned short int rec_len;  /* number of elements in each record */
} HEADER;

/* defines for data type used ind data file header and structure */

MATRIX *Matrix_Read();
DATA_FILE *Matrix_Write();
DATA_FILE *Open_Read();
DATA_FILE *Open_Write();
void Read_Record();
void Write_Record();
MATRIX *Matrix_Allocate();
void Matrix_Free();

/*********************  End of the Header ************************/ 

/* bpr.c program for operating a two-layer neural network           */

double **Input_Pattern;
double **Target_Pattern;
double **IH_Wts;
double **HO_Wts;
double *Onode_Thr;
double *Hnode_Thr;
double **Onode_Act;
double *Hnode_Act;
double *Estimate;

/*------------ 1_HL Backpropagation Networks for Training ------------*/

main ()
{

int np;              /* Index variable for number of patterns      */

FILE *Estimate_fp;

Read_Input_Patterns();

if (PRINT_TARGET) {
   Read_Target_Patterns();
}

if (NORMAL_WS) { 
    Normalize_Inputs_Patterns_Within();
}

if (NORMAL_AS) { 
    Normalize_Inputs_Patterns_Across();
}
Read_Network_Weights();
Read_Node_Thresholds(); 
Initialize_Network_Activations();
Estimate_fp = Open_Estimation_File_To_Write();
for(np=0; np< NPAT; ++np) {
    Compute_Hidden_Layer_Activation(np);
    Compute_Output_Layer_Activation(np);
}
Denormalize_Output_Layer_Activation(); 
Write_Asci_Estimates(Estimate_fp);
fclose(Estimate_fp);

if(REPLICATIONS > 1){
    Format_Asci_Estimates();
}

}
/*-------------------------- End of the main() -----------------------*/


/*----------------------- Read_Input_Patterns() ----------------------*/

void Read_Input_Patterns()
{

    char *input_file;
    FILE *input_fp;
    int i, j, k; 
    char data_element[20];
    double r;

    input_fp = fopen(INPUTS,"r");

/*
    printf("\n\n\n\n    ");
    input_file = Get_String("Test input pattern file name (e.g. input.dat)");
    input_fp = fopen(input_file,"r");
*/

/*
 *  input_file = Vector_Allocate_Char(80);
 *  strcpy(input_file, "input.dat");
 *  input_fp = fopen(input_file,"r");
 */

    if(!input_fp) {
        printf("\n  %s not found \n", input_file);
        printf("\n Press any character to exit...");
        getchar();
        exit(1);
    }   

    Input_Pattern = Matrix_Allocate_Double(NPAT,INODES);

    for(i=0; i< CORENPAT; i++) {
        for(j=0; j< INODES; j++) {
            fscanf(input_fp," %s", data_element);
            Input_Pattern[i][j] = (double) atof(data_element);
        }
        fscanf(input_fp,"\n");
    }
    for(i=CORENPAT; i< NPAT; i++) {
        for(j=0; j< INODES; j++) {
            k = (i % CORENPAT);
            if (j == (NF_VAR-1)){
               Input_Pattern[i][j] = Input_Pattern[k][j];
            } else {
               r = (double) ((double) rand()/(double) RAND_MAX);
               Input_Pattern[i][j] = MinLimit[j] + r*(MaxLimit[j]-MinLimit[j]);
            }
        }
    }
    fclose(input_fp);
    return;
}

/*--------------- Normalize_Inputs_Patterns_Within() -----------------*/
          
void Normalize_Inputs_Patterns_Within()
{
    int np, i;
    double Max;
    double Min;

    for(i=0;i<INODES;i++) {
        Max=Input_Pattern[0][i];
        Min=Input_Pattern[0][i];
        for(np=0;np<NPAT;np++) {
            if(Input_Pattern[np][i] > Max){
                Max = Input_Pattern[np][i];
            }
            if(Input_Pattern[np][i]<Min){
                Min = Input_Pattern[np][i];
            }
        }
        for(np=0;np<NPAT;np++) {
           Input_Pattern[np][i] = (Input_Pattern[np][i] -  Min)/(Max - Min);
        }
     }
return;
}

/*----------------- Normalize_Inputs_Patterns_Across() ---------------*/
          
void Normalize_Inputs_Patterns_Across()
{
    int np, i;
    double sum;

    for(np=0;np<NPAT;np++) {
        sum=0.0;
        for(i=0;i<INODES;i++) {
            sum += Input_Pattern[np][i]*Input_Pattern[np][i];
        }
        for(i=0;i<INODES;i++) {
            Input_Pattern[np][i]=Input_Pattern[np][i]/sqrt(sum);
        }
    }
    return;
}

/*---------------------- Read_Target_Patterns() ----------------------*/

void Read_Target_Patterns()
{

    char *target_file;
    FILE *target_fp;
    int i, j; 
    char data_element[20];
 
 
    target_fp = fopen (TARGETS,"r");
 /*
    printf("\n\n\n\n    ");
    target_file=Get_String("Test target pattern file name (e.g. target.dat)");
    target_fp = fopen (target_file,"r");
 */
  
/*
 *  target_file = Vector_Allocate_Char(80);
 *  strcpy(target_file, "target.dat");
 *  target_fp = fopen(target_file,"r");
 */
   
    if(!target_fp) {
        printf("\n  %s not found \n", target_file);
        exit(1);
    }   
    
    Target_Pattern = Matrix_Allocate_Double(NPAT,ONODES);
 
    for(i=0; i< NPAT; i++) {
        for(j=0; j< ONODES; j++) {
            fscanf(target_fp," %s", data_element);
            Target_Pattern[i][j] = (double) atof(data_element);
        }
        fscanf(target_fp,"\n");
    }   
    fclose(target_fp);
    return;
}

/*----------------------- Read_Network_Weights () --------------------*/

void Read_Network_Weights()
{
    MATRIX_DOUBLE *MDIH;
    MATRIX_DOUBLE *MDHO;
/*  char *wts_input_file;   */

/*
 *   Clear_Screen;
 *   Move_To(10,5);
 *   wts_input_file=
 *   Get_String("Enter input-hidden weights file to read");
 *   MDIH = (MATRIX_DOUBLE *) Matrix_Read(wts_input_file);
 */

    MDIH = (MATRIX_DOUBLE *) Matrix_Read("ih_wts.dat");

    if(MDIH->element_size != sizeof(double)
            || MDIH->rows != HNODES || MDIH->cols != INODES ) {
        printf("\nWeights read from ih_wts.dat are inconsistant\n");
        exit(1);
    }
    IH_Wts = Matrix_Allocate_Double(HNODES, INODES);
    ASSIGN_MAT(MDIH->ptr,IH_Wts,HNODES,INODES,double,double);
    Matrix_Free((MATRIX_DOUBLE *) MDIH);
   
/*
 *   Clear_Screen;
 *   Move_To(10,5);
 *   wts_input_file=
 *   Get_String("Enter hidden-ouput weights file to read");
 *   MDHO = (MATRIX_DOUBLE *) Matrix_Read(wts_input_file);
 */

    MDHO = (MATRIX_DOUBLE *) Matrix_Read("ho_wts.dat");

    if(MDHO->element_size != sizeof(double)
            || MDHO->rows != ONODES || MDHO->cols != HNODES ) {
        printf("\nWeights read from ho_wts.dat are inconsistant\n");
        exit(1);
    }
    HO_Wts = Matrix_Allocate_Double(ONODES, HNODES);
    ASSIGN_MAT(MDHO->ptr,HO_Wts,ONODES,HNODES,double,double);
    Matrix_Free((MATRIX_DOUBLE *) MDHO);
    return;
}

/*----------------------- Read_Node_Thresholds () --------------------*/

void Read_Node_Thresholds()
{
    MATRIX_DOUBLE *MDO;
    MATRIX_DOUBLE *MDH;
/*  char *thr_input_file;   */

/*
 *   Clear_Screen;
 *   Move_To(10,5);
 *   thr_input_file=
 *   Get_String("Enter output nodes threshold file to read");
 *   MDO = (MATRIX_DOUBLE *) Matrix_Read(thr_input_file);
 */

    MDO = (MATRIX_DOUBLE *) Matrix_Read("o_thr.dat");

    if(MDO->element_size != sizeof(double)
            || MDO->rows != 1 || MDO->cols != ONODES ) {
        printf("\nThresholds read from o_thr.dat are inconsistant\n");
        exit(1);
    }
    Onode_Thr = Vector_Allocate_Double(ONODES);
    ASSIGN(MDO->ptr[0],Onode_Thr,ONODES,double,double);
    Matrix_Free((MATRIX_DOUBLE *) MDO);

/*
 *   Clear_Screen;
 *   Move_To(10,5);
 *   thr_input_file=Get_String("Enter hidden-ouput weight file");
 *   MDH = (MATRIX_DOUBLE *) Matrix_Read(thr_input_file);
 */

    MDH = (MATRIX_DOUBLE *) Matrix_Read("h_thr.dat");

    if(MDH->element_size != sizeof(double)
            || MDH->rows != 1 || MDH->cols != HNODES ) {
        printf("\nThresholds read from h_thr.dat are inconsistant\n");
        exit(1);
    }
    Hnode_Thr = Vector_Allocate_Double(HNODES);
    ASSIGN(MDH->ptr[0],Hnode_Thr,HNODES,double,double);
    Matrix_Free((MATRIX_DOUBLE *) MDH);
    return;
}

/*------------------ Initialize_Network_Activations() ----------------*/

void Initialize_Network_Activations()
{
    int np,j;

    Onode_Act = Matrix_Allocate_Double(NPAT,ONODES);
    
    for(np=0; np< ONODES; np++)  {
        for(j=0; j< ONODES; j++)  {
            Onode_Act[np][j] = 0.0;
        }
    }

    Hnode_Act = Vector_Allocate_Double(HNODES);
    for(j=0; j< HNODES; j++)  {
        Hnode_Act[j] = 0.0;
    }
    return;
}

/*----------------- Open_Estimatoin_File_To_Write() ------------------*/

FILE *Open_Estimation_File_To_Write()
{
    char *est_file;
    FILE *est_fp;


    est_fp = fopen(RESULTS,"w");
/*
    printf("\n\n\n\n    ");
    est_file = Get_String("Enter estimation output file (e.g. result.out)");
    est_fp = fopen(est_file,"w");
 */
 
/*    
 *    est_file = Vector_Allocate_Char(20);
 *    strcpy(est_file,"result.out");
 *    est_fp = fopen(est_file,"w");
 */
     if(!est_fp) {
          printf("\n  %s cann't be openned \n", est_file);
     exit(1);
     }
     return est_fp;
}

/*---------------- Compute_Hidden_Layer_Activation() -----------------*/

void Compute_Hidden_Layer_Activation(np)
int np;
{
    double sum;
    double netsum;
    int i, j;

     for(j=0; j< HNODES; j++) {
         sum = 0.0;
         for(i=0; i< INODES; i++) {
             sum += IH_Wts[j][i]*Input_Pattern[np][i];
         }
         netsum = sum + Hnode_Thr[j];
         if(fabs(netsum) > SATURATION) {
             if(netsum < 0.0) {
                 netsum = -SATURATION;
             }
             else {
                 netsum = SATURATION;
             }
         }
/*
 *        if(ACT_FUNC==0) {
 *            Hnode_Act[j] = 1.0/(1.0+exp(-SLOPE2*netsum));
 *        }
 *        else if(ACT_FUNC==1) {
 *            Hnode_Act[j] = (1.0-exp(-SLOPE2*netsum))
 *                                      /(1.0+exp(-SLOPE2*netsum));
 *        }
 */
          Hnode_Act[j] = (1.0-exp(-SLOPE2*netsum))
                                        /(1.0+exp(-SLOPE2*netsum));
     }
     return;
}
     
/*----------------- Compute_Output_Layer_Activation() ----------------*/

void Compute_Output_Layer_Activation(np)
int np;
{
    double sum;
    double netsum;
    int i, j;

     for(j=0; j< ONODES; j++) {
         sum = 0.0;
         for(i=0; i< HNODES; i++) {
             sum += HO_Wts[j][i]*Hnode_Act[i];
         }
         netsum = sum + Onode_Thr[j];
         if(fabs(netsum) > SATURATION) {
             if(netsum < 0.0) {
                 netsum = -SATURATION;
             }
             else {
                 netsum = SATURATION;
             }
         }

         if(ACT_FUNC==0) {
             Onode_Act[np][j] = 1.0/(1.0+exp(-SLOPE1*netsum));
         }
         else if(ACT_FUNC==1) {
             Onode_Act[np][j] = (1.0-exp(-SLOPE1*netsum))
                                        /(1.0+exp(-SLOPE1*netsum));
         }
     }
return;
}

/*------------ Denormalize_Output_Layer_Activation() --------------*/

void Denormalize_Output_Layer_Activation() 
{
int np,j;
int pnit; 
double Max, Min;
FILE *Bpt2Bpr_fp;
int inodes,onodes,hnodes;
int act_func,normal_ws,normal_as; 

    Bpt2Bpr_fp = fopen("bpt2bpr.dat","r");

    if(!Bpt2Bpr_fp){
        printf("\n bpt2bpr.dat not found \n");
        exit(1);
    }

    for(j=0;j<ONODES;j++) {
       fscanf(Bpt2Bpr_fp,"%lf %lf\n", &Min, &Max);
       for(np=0;np<NPAT;np++) {
          if(ACT_FUNC) {
             Onode_Act[np][j]  = ((Onode_Act[np][j]+0.95)/1.9)*(Max - Min) + Min;
          }
          else {
             Onode_Act[np][j]  = ((Onode_Act[np][j]-0.05)/0.9)*(Max - Min) + Min;
          }
          if (Onode_Act[np][j] < Min) Onode_Act[np][j] = Min;
          if (Onode_Act[np][j] > Max) Onode_Act[np][j] = Max;
      }
   }

    fscanf(Bpt2Bpr_fp,"%d %d %d %d\n",&pnit,&inodes,&onodes,&hnodes);

    fscanf(Bpt2Bpr_fp,"%d %d %d",&act_func,&normal_ws,&normal_as); 

    if(inodes != INODES) {
        printf("\n Error:  INODES is not the same in bpt.c and bpr.c \n");
        printf("\n Press any key to exit ... \n");
        getchar();
        exit(1);
    }
    
    if(onodes != ONODES) {
        printf("\n Error:  ONODES is not the same in bpt.c and bpr.c \n");
        printf("\n Press any key to exit ... \n");
        getchar();
        exit(1);
    }
    
    if(hnodes != HNODES) {
        printf("\n Error:  HNODES is not the same in bpt.c and bpr.c \n");
        printf("\n Press any key to exit ... \n");
        getchar();
        exit(1);
    }
    
    if(act_func != ACT_FUNC) {
        printf("\n Error:  ACT_FUNC is not the same in bpt.c and bpr.c \n");
        printf("\n Press any key to exit ... \n");
        getchar();
        exit(1);
    }
    
    if(normal_ws != NORMAL_WS) {
        printf("\n Error:  NORMAL_WS is not the same in bpt.c and bpr.c \n");
        printf("\n Press any key to exit ... \n");
        getchar();
        exit(1);
    }
    
    if(normal_as != NORMAL_AS ){
        printf("\n Error:  NORMAL_AS is not the same in bpt.c and bpr.c \n");
        printf("\n Press any key to exit ... \n");
        getchar();
        exit(1);
    }
    
    fclose(Bpt2Bpr_fp);

return;
}

/*----------------------- Write_Asci_Estimates() ------------------*/

void Write_Asci_Estimates(est_fp)
FILE *est_fp;
{
    int np;
    int j;

    if(PRINT_TARGET) {
       // fprintf(est_fp,"No. \t Actual \t  Estimated  \t  Difference\n");
        fprintf(est_fp,"No.,Actual,Estimated,Difference\n");
    }
    else {
         //fprintf(est_fp,"No. \t Estimated\n");
         fprintf(est_fp,"No.,Estimated\n");
    }
    for(np=0; np< NPAT; ++np) {
       //fprintf(est_fp,"%d\t", np+1);
       fprintf(est_fp,"%d,", np+1);
       for(j=0; j< ONODES; j++) { 
          if(PRINT_TARGET) {
              // fprintf(est_fp,"% lf\t% lf\t% lf", Target_Pattern[np][j], Onode_Act[np][j], (Target_Pattern[np][j]-Onode_Act[np][j]));
              fprintf(est_fp,"% lf,% lf,% lf", Target_Pattern[np][j], Onode_Act[np][j], (Target_Pattern[np][j]-Onode_Act[np][j]));
          }
          else {
             // fprintf(est_fp,"% lf\t", Onode_Act[np][j]);
              fprintf(est_fp,"% lf", Onode_Act[np][j]);
          }
       }
       fprintf(est_fp," \n");
    }
    return;
}

/*---------------------- Format_Asci_Estimates() ------------------*/
void Format_Asci_Estimates() 
{
FILE *fmt_fp, *est_fp;
char fname[80], line[80];
int nnpat, num, np, r, n;
double estimate[NPAT];
double sum[CORENPAT], var[CORENPAT], org_input[CORENPAT]; 
double Mean[CORENPAT], StdDev[CORENPAT], Lcl[CORENPAT], Ucl[CORENPAT];
double x[NPAT];

    strcpy(fname,"fmt_");
    strcat(fname,RESULTS);
    fmt_fp = fopen(fname,"w");
    est_fp = fopen(RESULTS,"r");

    if(!fmt_fp) {
         printf("\n  %s cann't be openned \n", fname);
         printf("\n  Press any character to exit... \n");
         getchar();
         exit(1);
    }

    if(!est_fp) {
         printf("\n  %s cann't be openned \n", RESULTS);
         printf("\n  Press any character to exit... \n");
         getchar();
         exit(1);
    } 
    fgets(line, 70, est_fp);

    for(np=0;np< NPAT; np++){
        fscanf(est_fp,"%d,%lf\n", &num, &estimate[np]);
        x[np] = estimate[np] + Gaussian_Random_Number(M_MEAN, M_STDDEV);
    }

    nnpat = NPAT/REPLICATIONS;
    
    for(np=0;np < nnpat; np++){
        sum[np] = 0.0;
        for(n=0;n< REPLICATIONS; n++){
            sum[np] = sum[np] + x[(np+n*nnpat)];
        }
        Mean[np] = sum[np]/((double)REPLICATIONS);
    }

    for(np=0;np < nnpat; np++){
        var[np] = 0.0;
        for(n=0;n< REPLICATIONS; n++){
            var[np] = var[np] + (Mean[np]-x[(np+n*nnpat)])*
                                (Mean[np]-x[(np+n*nnpat)]);
        }
        StdDev[np] = sqrt(var[np]/((double)REPLICATIONS-1));
    }
    for(np=0;np < nnpat; np++){
        Lcl[np] = Mean[np] - 3.0*StdDev[np];
        Ucl[np] = Mean[np] + 3.0*StdDev[np];
    }

    for(np=0;np<nnpat;np++) {
        Input_Pattern[np][(NF_VAR-1)] =  MinLimit[(NF_VAR-1)]+
                                         (MaxLimit[(NF_VAR-1)] - MinLimit[(NF_VAR-1)])*
                                         Input_Pattern[np][(NF_VAR-1)];
    }

    fprintf(fmt_fp,"%s,%s,%s,%s,%s\n","Variable","Mean","StdDev","LCL","UCL");

    for(np=0;np< nnpat; np++){
        //fprintf(fmt_fp,"%d",np+1);
        fprintf(fmt_fp,"%lf,%lf,%lf,%lf,%lf",Input_Pattern[np][(NF_VAR-1)], Mean[np],
                                          StdDev[np], Lcl[np], Ucl[np]);
        // for(n=0;n< REPLICATIONS; n++){
        //     fprintf(fmt_fp,",%lf",x[(np+n*nnpat)]); 
        // }
        fprintf(fmt_fp,"\n");
    }
    fclose(fmt_fp);
    fclose(est_fp);
    return;
}


/* bplib.c is a library of procedures called in bpt.c and bpr.c */

/*------------------ Vector_Allocate_Char() -------------------*/

char *Vector_Allocate_Char(vec_size)
int vec_size;
{
	char *vec;

    vec = (char *) calloc((unsigned) vec_size, sizeof(char));
    if(!vec) {
	printf("\nBuffer allocation error\n");
	exit(1);
    }
	return vec;
}


/*------------------ Vector_Allocate_Float() -------------------*/

float *Vector_Allocate_Float(vec_size)
int vec_size;
{
	float *vec;

    vec = (float *) calloc((unsigned) vec_size, sizeof(float));
    if(!vec) {
	printf("\nBuffer allocation error\n");
	exit(1);
    }
	return vec;
}

/*------------------ Vector_Allocate_Double() -------------------*/

double *Vector_Allocate_Double(vec_size)
int vec_size;
{
	double *vec;

    vec = (double *) calloc((unsigned) vec_size, sizeof(double));
    if(!vec) {
	printf("\nBuffer allocation error\n");
	exit(1);
    }
	return vec;
}


/*----------------- Vector_Allocate_Int () -------------------*/

int *Vector_Allocate_Int(vec_size)
int vec_size;
{
	int *vec;

    vec = (int *) calloc((unsigned) vec_size, sizeof(int));
    if(!vec) {
	printf("\nBuffer allocation error\n");
	exit(1);
    }
	return vec;
}


/*------------------- Matrix_Allocate_Char () -------------------*/

char **Matrix_Allocate_Char (mat_rows, mat_cols)
int mat_rows, mat_cols;
{
	int i;
	char **mat;

    mat = (char **) calloc((unsigned) mat_rows, sizeof(char *));
    if(!mat) {
		printf("\nBuffer allocation error\n");
		exit(1);
    }
    for (i=0; i< mat_rows; i++) {
    	mat[i] = (char *) calloc((unsigned) mat_cols, sizeof(char));
    	if(!mat[i]) {
			printf("\nBuffer allocation error\n");
			exit(1);
    	}
	}
	return mat;
}		


/*------------------- Matrix_Allocate_Float () -------------------*/

float **Matrix_Allocate_Float(mat_rows, mat_cols)
int mat_rows, mat_cols;
{
	int i;
	float **mat;

    mat = (float **) calloc((unsigned) mat_rows, sizeof(float *));
    if(!mat) {
		printf("\nBuffer allocation error\n");
		exit(1);
    }
    for (i=0; i< mat_rows; i++) {
    	mat[i] = (float *) calloc((unsigned) mat_cols, sizeof(float));
    	if(!mat[i]) {
			printf("\nBuffer allocation error\n");
			exit(1);
    	}
	}
	return mat;
}		


/*------------------- Matrix_Allocate_Double () -------------------*/

double **Matrix_Allocate_Double(mat_rows, mat_cols)
int mat_rows, mat_cols;
{
	int i;
	double **mat;

    mat = (double **) calloc((unsigned) mat_rows, sizeof(double *));
    if(!mat) {
		printf("\nBuffer allocation error\n");
		exit(1);
    }
    for (i=0; i< mat_rows; i++) {
    	mat[i] = (double *) calloc((unsigned) mat_cols, sizeof(double));
    	if(!mat[i]) {
			printf("\nBuffer allocation error\n");
			exit(1);
    	}
	}
	return mat;
}		


/*------------------- Matrix_Allocate_Int () -------------------*/

int **Matrix_Allocate_Int(mat_rows, mat_cols)
int mat_rows, mat_cols;
{
	int i;
	int **mat;

    mat = (int **) calloc((unsigned) mat_rows, sizeof(int *));
    if(!mat) {
		printf("\nBuffer allocation error\n");
		exit(1);
    }
    for (i=0; i< mat_rows; i++) {
    	mat[i] = (int *) calloc((unsigned) mat_cols, sizeof(int));
    	if(!mat[i]) {
			printf("\nBuffer allocation error\n");
			exit(1);
    	}
	}
	return mat;
}		


/*----------------- Mat_Array_Allocate_Double () ----------------*/

double ***Mat_Array_Allocate_Double(mat_size, mat_rows, mat_cols)
int mat_size, mat_rows, mat_cols;
{
	int i, j;
	double ***mat;

    mat = (double ***) calloc((unsigned) mat_size, sizeof(double **));
    if(!mat) {
		printf("\nBuffer allocation error\n");
		exit(1);
    }
    for (j=0; j< mat_size; j++) {
    	mat[j] = (double **)calloc((unsigned)mat_rows,sizeof(double *));
    	if(!mat[j]) {
			printf("\nBuffer allocation error\n");
			exit(1);
    	}
    	for (i=0; i< mat_rows; i++) {
    		mat[j][i]=(double *)calloc((unsigned)mat_cols,
													sizeof(double));
    		if(!mat[j][i]) {
				printf("\nBuffer allocation error\n");
				exit(1);
    		}
		}
	}
return mat;
}


/***********************************************************************

GET.C - Source code for user input functions

    Get_String      get string from user with prompt
    Get_Int         get integer from user with prompt and range
    Get_Float       get float from user with prompt and range
    Get_Double      get double from user with prompt and range

***********************************************************************/


/***********************************************************************
Get_String - get string from user with prompt

Return pointer to string of input text, prompts user with string
passed by caller.  Indicates error if string space could not be
allocated.  Limited to 80 char input.

char *Get_String(char *prompt_string)

    prompt_string  string to prompt user for input

***********************************************************************/

char *Get_String(title_string)
    char *title_string;
{
    char *alpha;                            /* result string pointer */

    alpha = (char *) malloc(80);
    if(!alpha) {
        printf("\nString allocation error in Get_String\n");
        exit(1);
    }
    printf("\n %s: ",title_string);
    gets(alpha);

    return(alpha);
}
/***********************************************************************

Get_Int - get integer from user with prompt and range

Return integer of input text, prompts user with prompt string
and range of values (upper and lower limits) passed by caller.

int Get_Int(char *title_string,int low_limit,int up_limit)

    title_string  string to prompt user for input
    low_limit     lower limit of acceptable input (int)
    up_limit      upper limit of acceptable input (int)

***********************************************************************/

int Get_Int(title_string,low_limit,up_limit)
    char *title_string;
    int low_limit,up_limit;
{
    int i,error_flag;
    char *Get_String();             /* get string routine */
    char *cp,*endcp;                /* char pointer */
    char *stemp;                    /* temp string */

/* check for limit error, low may equal high but not greater */
    if(low_limit > up_limit) {
        printf("\nLimit error, lower > upper\n");
        exit(1);
    }

/* make prompt string */
    stemp = (char *) malloc(strlen(title_string) + 60);
    if(!stemp) {
        printf("\nString allocation error in Get_Int\n");
        exit(1);
    }
    sprintf(stemp,"%s [%d...%d]",title_string,low_limit,up_limit);

/* get the string and make sure i is in range and valid */
    do {
        cp = Get_String(stemp);
        i = (int) strtol(cp,&endcp,10);
        error_flag = (cp == endcp)||(*endcp != '\0'); /*detect errors */
        free(cp);                                 /*free string space */
    } while(i < low_limit || i > up_limit || error_flag);

/* free temp string and return result */
    free(stemp);
    return(i);
}


/***********************************************************************

Get_Float - get float from user with prompt and range

Return double of input text, prompts user with prompt string
and range of values (upper and lower limits) passed by caller.

double Get_Float(char *title_string,float low_limit,float up_limit)

    title_string  string to prompt user for input
    low_limit     lower limit of acceptable input (float)
    up_limit      upper limit of acceptable input (float)

***********************************************************************/

float Get_Float(title_string,low_limit,up_limit)
    char *title_string;
    float low_limit,up_limit;
{
    float x;
    int error_flag;
    char *Get_String();             /* get string routine */
    char *cp,*endcp;                /* char pointer */
    char *stemp;                    /* temp string */

/* check for limit error, low may equal high but not greater */
    if(low_limit > up_limit) {
        printf("\nLimit error, lower > upper\n");
        exit(1);
    }

/* make prompt string */
    stemp = (char *) malloc(strlen(title_string) + 80);
    if(!stemp) {
        printf("\nString allocation error in Get_Float\n");
        exit(1);
    }

    sprintf(stemp,"%s [%1.2g...%1.2g]",title_string,low_limit,up_limit);
    
/* get the string and make sure x is in range */
    do {
        cp = Get_String(stemp);
        x = (float)strtod(cp,&endcp);
        error_flag = (cp == endcp)||(*endcp != '\0'); /*detect errors */
        free(cp);                                /* free string space */
    } while(x < low_limit || x > up_limit || error_flag);

/* free temp string and return result */
    free(stemp);
    return(x);
}

/***********************************************************************

Get_Double - get double from user with prompt and range

Return double of input text, prompts user with prompt string
and range of values (upper and lower limits) passed by caller.

double Get_Double(char *title_string,double low_limit,double up_limit)

    title_string  string to prompt user for input
    low_limit     lower limit of acceptable input (double)
    up_limit      upper limit of acceptable input (double)

***********************************************************************/

double Get_Double(title_string,low_limit,up_limit)
    char *title_string;
    double low_limit,up_limit;
{
    double x;
    int error_flag;
    char *Get_String();             /* get string routine */
    char *cp,*endcp;                /* char pointer */
    char *stemp;                    /* temp string */

/* check for limit error, low may equal high but not greater */
    if(low_limit > up_limit) {
        printf("\nLimit error, lower > upper\n");
        exit(1);
    }

/* make prompt string */
    stemp = (char *) malloc(strlen(title_string) + 80);
    if(!stemp) {
        printf("\nString allocation error in Get_Float\n");
        exit(1);
    }

    sprintf(stemp,"%s [%1.2g...%1.2g]",title_string,low_limit,up_limit);
    
/* get the string and make sure x is in range */
    do {
        cp = Get_String(stemp);
        x = strtod(cp,&endcp);
        error_flag = (cp == endcp)||(*endcp != '\0'); /*detect errors */
        free(cp);                                /* free string space */
    } while(x < low_limit || x > up_limit || error_flag);

/* free temp string and return result */
    free(stemp);
    return(x);
}


/*------------------------- Error_Msg () --------------------------*/

void Error_Msg(error_msg)
char error_msg[];
{
	printf("\n %s \n",error_msg);
	printf("\n...now exiting to system...\n");
	exit(1);
}

/***********************************************************************

DATAIO.C - Source code for data Read and write functions

   Matrix_Read             read one matrix
   Matrix_Write            write one matrix
   Matrix_Free             free matrix area and MATRIX structure
   Open_Read               open data file to be read
   Open_Write              create header and open data file for write
   Read_Record             read one record
   Write_Record            write one record
   Matrix_Allocate   allocates the necessary memory space for a matrix
   Matrix_Free       frees the allocated memory sapce for a matrix

***********************************************************************/

/***********************************************************************

Matrix_Read - open a data file and read it in as a matrix

Open file using file_name and returns pointer to MATRIX structure.
Allocation errors or improper type causes a call to exit(1).
A bad file name returns a NULL pointer.

MATRIX *Matrix_Read(char *file_name)

***********************************************************************/

MATRIX *Matrix_Read(file_name)
    char *file_name;
{
    MATRIX *matrix_Allocate();
    DATA_FILE *data_info;
    MATRIX *A;
    int i,mat_size,length;
    double *buf;          /* input buffer to read data in */

    data_info = Open_Read(file_name);
    if(!data_info) return(NULL);     /* bad filename case */

/* determine size of matrix (int, float or double) */
    switch(data_info->type) {
	   	case FLOAT:
	    	mat_size = sizeof(float);
	    	break;
		case DOUBLE:
	    	mat_size = sizeof(double);
		  	break;
    }

/* allocate matrix */
    A = Matrix_Allocate(data_info->records,data_info->rec_len,mat_size);
			    
/* row length */
    length = data_info->rec_len;

/* allocate input buffer area cast to double for worst case alignment */
    buf = (double *) calloc(length,data_info->element_size);
    if(!buf) {
	printf("\nBuffer allocation error in Matrix_Read\n");
	exit(1);
    }

/* read each row and translate */
    for(i = 0 ; i < data_info->records ; i++) {
		Read_Record((char *)buf,data_info);

	   	switch(data_info->type) {
		  	case FLOAT:
				ASSIGN(buf,A->ptr[i],length,float,float)
				break;
	    	case DOUBLE:
				ASSIGN(buf,A->ptr[i],length,double,double)
				break;
	    }
    }

/* free the data info structure and close file since not used again */
    free(data_info->name);
    fclose(data_info->fp);
    free((char *)data_info);

    return(A);
}

/***********************************************************************

Matrix_Write - open a data file and write out a matrix

Writes matrix of the same type as given in the MATRIX structure
(int, float or double) to file_name.

Returns the DATA_FILE structure for use by other disk I/O
routines such as write_trailer.
				    
Allocation errors or improper type causes a call to exit(1).
Disk open error returns a NULL pointer.

DATA_FILE *Matrix_Write(MATRIX *A,char *file_name)

***********************************************************************/

DATA_FILE *Matrix_Write(A,file_name)
    MATRIX *A;
    char *file_name;
{
    DATA_FILE *data_info;
    int i,type;

    switch(A->element_size) {
	  case sizeof(float) :
	    type = FLOAT;
	    break;
	  case sizeof(double) :
	    type = DOUBLE;
	    break;
	  default:
	    printf("\nError in MATRIX structure\n");
	    exit(1);
    }

    data_info = Open_Write(file_name,type,A->rows,A->cols);
/* if no open return NULL pointer */
    if(!data_info) return(NULL);
			
/* write each row */
    for(i = 0 ; i < data_info->records ; i++)
	   Write_Record(A->ptr[i],data_info);

    return(data_info);
}


/***********************************************************************

Open_Read - open a data file for read

Returns a pointer to a DATA_FILE structure allocated by the
function and opens file_name.

Allocation errors or improper type causes a call to exit(1).
A bad file_name returns a NULL pointer.

DATA_FILE *Open_Read(char *file_name)

***********************************************************************/

DATA_FILE *Open_Read(file_name)
    char *file_name;            /* file name string */
{
    DATA_FILE *data_info;
    int status;

/* allocate the data file structure */

    data_info = (DATA_FILE *) malloc(sizeof(DATA_FILE));
    if(!data_info) {
	printf("\nError in Open_Read: structure allocation, file %s\n",
		    file_name);
	exit(1);
    }

/* open file for binary read and update */
    data_info->fp = fopen(file_name,"r+b");
    if(!data_info->fp) {
	printf("\nError opening %s in Open_Read\n",file_name);
	return(NULL);
    }

/* copy and allocate file name string for the DATA_FILE structure */
    data_info->name = malloc(strlen(file_name) + 1);
    if(!data_info->name) {
	printf("\nUnable to allocate file_name string in Open_Read\n");
	exit(1);
    }
    strcpy(data_info->name,file_name);

/* read in header from file */
    status = fread((char *)data_info,sizeof(HEADER),1,data_info->fp);
    if(status != 1) {
	printf("\nError reading header of file %s\n",file_name);
	exit(1);
    }

/* return pointer to DATA_FILE structure */
    return(data_info);
}

/***********************************************************************

Open_Write - open a data file for write

Returns a pointer to a DATA_FILE structure allocated by the function.
Allocation errors or improper type causes a call to exit(1).
A bad file name returns a NULL pointer.

DATA_FILE *Open_Write(char *file_name,int type,int records,int rec_len)

    file_name       pointer to file name string
    type            type of data (1-3 specified in defines)
    records         number of records of data to be written
    rec_len         number of elements in each record

***********************************************************************/

DATA_FILE *Open_Write(file_name,type,records,rec_len)
    char *file_name;              /* file name string */
    int type;                     /* data type 1-3    */
    unsigned short int records;   /* number of records to be written */
    unsigned short int rec_len;   /* elements in each record */
{
    DATA_FILE *data_info;
    int status;

/* allocate the data file structure */
    data_info = (DATA_FILE *) malloc(sizeof(DATA_FILE));
    if(!data_info) {
	printf("\nError in Open_Write: structure allocation, file %s\n",
		    file_name);
	exit(1);
    }

/* set the basics */
    data_info->type = (unsigned char)type;
    data_info->records = records;
    data_info->rec_len = rec_len;

/* set element size from data type */
    switch(type) {
	   case 1:
		 data_info->element_size = sizeof(float);
	     break;
	   case 2:
		 data_info->element_size = sizeof(double);
	     break;
	   case 3:
	     data_info->element_size = sizeof(short int);
	     break;
	   default:
	     printf("\nUnsupported data type, file %s\n",file_name);
	     exit(1);
    }

/* open file for binary write */
    data_info->fp = fopen(file_name,"wb");
    if(!data_info->fp) {
	printf("\nError opening %s in Open_Write\n",file_name);
	return(NULL);
    }

/* copy and allocate file name string for the DATA_FILE structure */
    data_info->name = malloc(strlen(file_name) + 1);
    if(!data_info->name) {
	printf("\nUnable to allocate file_name string in Open_Write\n");
	exit(1);
    }
    strcpy(data_info->name,file_name);

/* write header to file */
    status = fwrite((char *)data_info,sizeof(HEADER),1,data_info->fp);
    if(status != 1) {
	printf("\nError writing header of file %s\n",file_name);
	exit(1);
    }

/* return pointer to DATA_FILE structure */
    return(data_info);
}

/***********************************************************************

Read_Record - read one record of data file

Exits if a read error occurs or if the DATA_FILE structure is invalid.
    
void Read_Record(char *ptr,DATA_FILE *data_info)

    ptr        pointer to previously allocated memory to put data
    data_info   pointer to data file structure

***********************************************************************/

void Read_Record(ptr,data_info)
    char *ptr;                  /* pointer to some type of data */
    DATA_FILE *data_info;
{
    int status;

    if(!data_info) {
	printf("\nError in DATA_FILE structure passed to Read_Record\n");
	exit(1);
    }

    status = fread(ptr,data_info->element_size,
			    data_info->rec_len,data_info->fp);
    if(status != data_info->rec_len) {
	printf("\nError in Read_Record, file %s\n",data_info->name);
	exit(1);
    }
}

/***********************************************************************

Write_Record - write one record of data

Exits if write error occurs or if the DATA_FILE structure is invalid.

void Write_Record(char *ptr,DATA_FILE *data_info)

    ptr        pointer to data to write to disk (type in data_info)
    data_info   pointer to data file structure

***********************************************************************/

void Write_Record(ptr,data_info)
    char *ptr;                  /* pointer to some type of data */
    DATA_FILE *data_info;
{
    int status;

    if(!data_info) {
	printf("\nError in DATA_FILE structure passed to Write_Record\n");
	exit(1);
    }

    status = fwrite(ptr,data_info->element_size,
			     data_info->rec_len,data_info->fp);
    if(status != data_info->rec_len) {
	printf("\nError Write_Record, file %s\n",data_info->name);
	exit(1);
    }
}


/* structure used by all matrix routines */

/*
typedef struct {
    int element_size;
    unsigned int rows;
    unsigned int cols;
    char **ptr;
               } MATRIX;
*/

/***********************************************************************

MATRIX.C - Source code for matrix functions

Matrix_Allocate   allocates the necessary memory space for a matrix
Matrix_Free       frees the allocated memory sapce for a matrix

***********************************************************************/

/***********************************************************************

Matrix_Allocate - Make matrix of given size (rows x cols) and type

The type is given by element_size (1 = floats, 2 = doubles, 3 = ints ) 
Exits if the matrix could not be allocated.

    MATRIX *Matrix_Allocate(int rows,int cols,int element_size)

***********************************************************************/

MATRIX *Matrix_Allocate(rows,cols,element_size)
    int rows,cols,element_size;
{
    int i;
    MATRIX *A;

/* allocate the matrix structure */
    A = (MATRIX *) calloc(1,sizeof(MATRIX));
    if(!A) {
        printf("\nERROR in matrix structure allocate\n");
        exit(1);
    }

/* set up the size as requested */
    A->rows = rows;
    A->cols = cols;
    A->element_size = element_size;

/* try to allocate the request */
    switch(element_size) {
   		case sizeof(float): {    /* float matrix */
            float **float_matrix;
            float_matrix = (float **)calloc(rows,sizeof(float *));
            if(!float_matrix) {
                printf("\nError making pointers in %dx%d float matrix\n"
                            ,rows,cols);
                exit(1);
            }
            for(i = 0 ; i < rows ; i++) {
                float_matrix[i] = (float *)calloc(cols,sizeof(float));
                if(!float_matrix[i]) {
                    printf("\nError making row %d in %dx%d float matrix\n",i,rows,cols);
                    exit(1);
                }
            }
            A->ptr = (char **)float_matrix;
            break;
        }
        case sizeof(double): {   /* double matrix */
            double **double_matrix;
            double_matrix = (double **)calloc(rows,sizeof(double *));
            if(!double_matrix) {
                printf("\nError making pointers in %dx%d double matrix\n"
                            ,rows,cols);
                exit(1);
            }
            for(i = 0 ; i < rows ; i++) {
                double_matrix[i] = (double *)calloc(cols,sizeof(double));
                if(!double_matrix[i]) {
                    printf("\nError making row %d in %dx%d double matrix\n",i,rows,cols);
                    exit(1);
                }
            }
            A->ptr = (char **)double_matrix;
            break;
        }
	   case sizeof(short): {    /* integer matrix */
            short **int_matrix;
            int_matrix = (short **)calloc(rows,sizeof(short *));
            if(!int_matrix) {
                printf("\nError making pointers in %dx%d int matrix\n"
                            ,rows,cols);
                exit(1);
            }
            for(i = 0 ; i < rows ; i++) {
                int_matrix[i] = (short *)calloc(cols,sizeof(short));
                if(!int_matrix[i]) {
                    printf("\nError making row %d in %dx%d int matrix\n"
                            ,i,rows,cols);
                    exit(1);
                }
            }
            A->ptr = (char **)int_matrix;
            break;
        }
        default:
            printf("\nERROR in Matrix_Allocate: unsupported type\n");
            exit(1);
    }
    return(A);
}


/***********************************************************************

Matrix_Free - Free all elements of matrix

Frees the matrix data (rows and cols), the matrix pointers and
the MATRIX structure.

Error message and exit if improper structure is
passed to it (null pointers or zero size matrix).

    void Matrix_Free(MATRIX *A)

***********************************************************************/

void Matrix_Free(A)
    MATRIX *A;
{
    unsigned int i;
    char **a;

/* check for valid structure */
    if(!A || !A->ptr || !A->rows || !A->cols) {
        printf("\nERROR: invalid structure passed to Matrix_Free");
        exit(1);
    }

/* a used for freeing */
    a = A->ptr;

/* free each row of data */
    for(i = 0 ; i < (A->rows) ; i++) free(a[i]);

/* free each row pointer */
    free((char *)a);
    a = NULL;           /* set to null for error */

/* free matrix structure */
    free((char *)A);
}
/*------------------- Gaussian_Random_Number() ---------------------*/

double Gaussian_Random_Number(mean, sdv)
double mean;
double sdv;
{
    double r1, r2, c1, value;

    r1 = (double) ((double) rand()/(double) RAND_MAX);
    r2 = (double) ((double) rand()/(double) RAND_MAX);
    c1 = (double)sdv*(sqrt((double)(-2.0*log((double)r1))));
    value = (double)mean+ c1*(cos(2.0*3.1416*r2));
    return value;
}

