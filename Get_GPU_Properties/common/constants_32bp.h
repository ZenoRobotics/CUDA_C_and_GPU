// File:  constants_32bp.h

// Constants
#define  NUM_OF_SEQUENCES_IN_CACHE             16 * 1024 //8 * 1024     // Number of database sequences stored in cache
#define  NUM_OF_ENTRIES_IN_PTR_ARRAYS          8 * 1024  //4 * 1024  
#define  HASH_MASK                             0x1fff        //Correlates to TABLE_LENGTH
#define  NUM_OF_LUTS_USED                      12            // Number of parallel LUTs used to find first occurrence
#define  NUM_OF_BPS_PER_QRY_SEQ                32            // Particular to the current algorithm
#define  X_MER_SIZE                            NUM_OF_BPS_PER_QRY_SEQ
#define  NUM_OF_BITS_FOR_QRY_SEQ_ID            16
#define  NUM_OF_BITS_FOR_DB_SEQ_ID             16
#define  NUM_OF_BITS_FOR_TOTAL_RSLT_WORD       16          // 13 bits for 1st occurrence offset + 3'b000
#define  MAX_HITS                              300         // Max number of hits/ptrs per bin recorded
#define  NUM_OF_BITS_PER_NUCLEOTIDE            2
#define  NUM_OF_BYTES_PER_WORD                 4

#define  QRY_SEGMENT_ID_INDICATOR              0x80000000  // Upper 3 bits of the 32 bit data result indicates that 
                                                           // the Qry segment ID # can be found in the lower 29 bits
#define  DB_SEGMENT_ID_INDICATOR               0xa0000000  // Upper 3 bits of the 32 bit data result indicates that  
                                                           // the DB segment ID # can be found in the lower 29 bits
#define  ADDITION_SEARCH_REQD_INDICATOR        0xc0000000  // Upper 3 bits of the 32 bit data result indicates that
                                                           // additional search required because # of unique matches is
                                                           // greater than # of unique lookup brams in FPGA. Seq Id is
                                                           // located in the lower 16 bits.
 
//Visual/Analyzed Report Processing Constanst

#define  SHOT_GUN_OVERSAMPLE_FACTOR                   5     // Temporary variable.  This value will be set and passed in main scripts.
#define  NUM_OF_WORDS_PER_MATCH                       4     // Number of 32 bit words per match
#define  OVERSAMPLE_CUSION_FACTOR                     5     // Extra entries factor above oversample factor
#define  NUM_OF_RSLT_RECORD_ENTRIES_PER_DB_OFFSET     SHOT_GUN_OVERSAMPLE_FACTOR*NUM_OF_WORDS_PER_MATCH*OVERSAMPLE_CUSION_FACTOR 

// Other
//create boolean logic
#ifndef BOOLEAN
  typedef int bool;
#endif
#define FALSE 0
#define TRUE  1

