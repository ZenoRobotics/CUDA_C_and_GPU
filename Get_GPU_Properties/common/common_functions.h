// File:  common_functions.h


//Shared Functions

long skip_header_data(FILE *);
int power(int , int );
unsigned int hash_algorithm(unsigned int , int, unsigned int );
unsigned int hash_algorithm_32bp(long long , int, unsigned int);
long long sequence_reverse_complement(long long , int );
bool check_for_valid_nucleotide(char);
unsigned int nucleotide_to_uint(char);
long long bp_string_to_uint(char *string, int x_mer_size); 


long skip_header_data(FILE *fin)  {
   char  hdr_string[120];
   char  nucleotide;
   long  current_file_addr = 0;
   bool  DONE = FALSE;

   current_file_addr = ftell(fin);  // Get current file position

   while (!DONE) {
      if (fscanf(fin, "%c", &nucleotide) <= 0) 
         DONE = TRUE;
      else if (check_for_valid_nucleotide(nucleotide))
         DONE = TRUE;
      else  {
          fgets(hdr_string , 120 , fin);
          current_file_addr = ftell(fin);
      }
   }
   
   return current_file_addr;
}

int power(int x, int y)  {
    int result = 1;
    int i;
    
    for (i=0; i < y; i++)  {
        result = result * x;
    }
    return result;
}

int log_2(int num)  {

   int  shift_cnt = 0;
   int  shift_val = 0;
   bool one_found = FALSE;

   shift_val = num;

   while(!one_found)  {
      shift_val = shift_val >> 1;
      shift_cnt += 1;
      if (shift_val == 1)  
         one_found = TRUE;

   }

   return shift_cnt;

}

unsigned int hash_algorithm(unsigned int base_x, int x_mer_size, unsigned int bit_mask)  {
   
    unsigned int hashed_array_addr = 0;
    unsigned int base_x_xor_upper  = 0;
    unsigned int base_x_xor_lower  = 0;
    
/*  //2x
    base_x_xor_upper  = ((base_x >> 0) ^ (base_x >> 9) ^ (base_x >> 19))   & bit_mask;
    base_x_xor_lower  = ((base_x >> 16) ^ (base_x >> 6) ^ (base_x >> 23))  & bit_mask;
*/   
/*  //4x
    base_x_xor_upper  = ((base_x >> 0) ^ (base_x >> 9) ^ (base_x >> 19) ^ (base_x >> 18))   & bit_mask;
    base_x_xor_lower  = ((base_x >> 16) ^ (base_x >> 6) ^ (base_x >> 23) )  & bit_mask;
*/
    base_x_xor_upper  = ((base_x >> 0) ^ (base_x >> 8) ^ (base_x >> 19) ^ (base_x >> 18))   & bit_mask;
    base_x_xor_lower  = ((base_x >> 16) ^ (base_x >> 9) ^ (base_x >> 23) ^ (base_x >> 18))  & bit_mask;

    hashed_array_addr = (base_x_xor_lower + base_x_xor_upper)  & bit_mask;
    

    return  hashed_array_addr;   //hashed_array_addr

}


unsigned int hash_algorithm_32bp(long long base_x, int x_mer_size, unsigned int bit_mask)  {
    
    unsigned int hashed_array_addr = 0;
    unsigned int base_x_xor_upper = 0;
    unsigned int base_x_xor_lower = 0;
    

    base_x_xor_upper  = ( ((base_x >> 44) & 0xf0f) ^ (base_x >> 0) ^ (base_x >> 9) )   & HASH_MASK;
    base_x_xor_lower  = ( ((base_x >> 51) & 0x5a5) ^ (base_x >> 16) ^ (base_x >> 30)  )  & HASH_MASK;

/*
    //original
    base_x_xor_upper  = ((base_x >> 46) ^ (base_x >> 0) ^ (base_x >> 9) ^ (base_x >> 39))   & bit_mask;
    base_x_xor_lower  = ((base_x >> 16) ^ (base_x >> 30) ^ (base_x >> 45) ^ (base_x >> 23))  & bit_mask;
*/
    hashed_array_addr = (base_x_xor_lower + base_x_xor_upper)  & HASH_MASK;

    return  hashed_array_addr;   
}


long long sequence_reverse_complement(long long orig_sequence,int x_mer_size) {

   int i;
   long long    rev_comp_seq     = 0;
   unsigned int comp_base[x_mer_size];     //5' Position of original sequence - complemented = index 0
                                           //3' Position of original sequence - complemented = index 15
  
   for (i=0; i < x_mer_size; i++) {
       comp_base[i] = ((orig_sequence >> (((x_mer_size - i) - 1) * 2)) & 3) ^ 3;
    }

    for (i= (x_mer_size - 1); i >= 0 ; i--) {
       rev_comp_seq = (comp_base[i] << (i * 2)) |  rev_comp_seq;
    }

   return rev_comp_seq;
}


bool check_for_valid_nucleotide(char nucleotide) {
   unsigned int uint_nuke = 99;

   switch (nucleotide) {
     case 'A' :  
        uint_nuke = 0;
     break;
              
     case 'a' :  
        uint_nuke = 0;
     break;
              
     case 'C' :
        uint_nuke = 1; 
     break;
              
     case 'c' :
        uint_nuke = 1; 
     break;
              
     case 'G' :
        uint_nuke = 2; 
     break;
              
     case 'g' :
        uint_nuke = 2; 
     break;
              
     case 'T' : 
        uint_nuke = 3;
     break;
              
     case 't' : 
        uint_nuke = 3;
     break;
              
     default : 
        uint_nuke = 99;
   }

   if (uint_nuke < 99)
     return TRUE;
   else
     return FALSE;
}

unsigned int nucleotide_to_uint(char nucleotide) {
   unsigned int uint_nuke;
   switch (nucleotide) {
     case 'A' :  
        uint_nuke = 0;
     break;
              
     case 'a' :  
        uint_nuke = 0;
     break;
              
     case 'C' :
        uint_nuke = 1; 
     break;
              
     case 'c' :
        uint_nuke = 1; 
     break;
              
     case 'G' :
        uint_nuke = 2; 
     break;
              
     case 'g' :
        uint_nuke = 2; 
     break;
              
     case 'T' : 
        uint_nuke = 3;
     break;
              
     case 't' : 
        uint_nuke = 3;
     break;
              
     default : 
        uint_nuke = 0;
   }
   return uint_nuke;
}


long long bp_string_to_uint(char *string, int x_mer_size) {
   long long uint_nuke;
   long long sequence = 0;
   int i;
   

   for (i=0; i < x_mer_size; i++)  {
      switch (string[i]) {
        case 'A' :  
           uint_nuke = 0;
        break;
              
        case 'a' :  
           uint_nuke = 0;
        break;
              
        case 'C' :
           uint_nuke = 1; 
        break;
              
        case 'c' :
           uint_nuke = 1; 
        break;
              
        case 'G' :
           uint_nuke = 2; 
        break;
              
        case 'g' :
           uint_nuke = 2; 
        break;
              
        case 'T' : 
           uint_nuke = 3;
        break;
              
        case 't' : 
           uint_nuke = 3;
        break;
              
        default : 
           uint_nuke = 0;
      }
      sequence = sequence | (uint_nuke << ((x_mer_size-1-i)*2));
   }
   return sequence;
}



