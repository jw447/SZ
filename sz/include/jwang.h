#include <sys/time.h>

#ifndef _CPU_timing
#define _CPU_timing 1
typedef struct {
  float compCost;             // time for SZ_compress_double_1D_MDQ
  float cfCost;               // time for curve-fitting
  float hitCost;              // time for curve-hit points
  float misCost;              // time for curve-missed points
  float cSDVCost;
  float uLCECost;
  float aEDCost;

  float treeCost;            // time for huffman tree
  float createTCost;         // time for creating huffman tree
  float buildTCost;          // time for building huffman tree
  float encodeTCost;         // time for encoding huffman tree

  int count_hit;
  int count_missed;
  double hit_ratio;
  int node_count;
  int Nelements;
  int qf;

  // openmp
  float hit_ratio_omp;

  double cfCost_omp;               // time for curve-fitting
  double hitCost_omp;              // time for curve-hit points
  double misCost_omp;              // time for curve-missed points
  double cSDVCost_omp;
  double uLCECost_omp;
  double aEDCost_omp;

  double tree0_omp;
  double tree1_omp;
  double tree2_omp;
  double tree3_omp;
  double tree4_omp;
  double tree5_omp;

  double costTree_omp;
  double costEncode_omp;

  //double treeCost_omp;
  //double createTCost_omp;
  //double buildTCost_omp;
  //double encodeTCost_omp;
} CPU_timing;

struct timeval compCostS, compCostE; // time for SZ_compress_double_1D_MDQ
struct timeval cfCostS, cfCostE;     // time for curve-fitting
struct timeval hitCostS, hitCostE;   // time for curve-hit points
struct timeval misCostS, misCostE;   // time for curve-missed points
struct timeval cSDVCostS, cSDVCostE;
struct timeval uLCECostS, uLCECostE;
struct timeval aEDCostS, aEDCostE;

struct timeval treeCostS, treeCostE; // time for huffman tree
struct timeval createTCostS, createTCostE; // time for creating huffman tree
struct timeval buildTCostS, buildTCostE;   // time for building huffman tree
struct timeval encodeTCostS, encodeTCostE; // time for encoding huffman tree

#endif
