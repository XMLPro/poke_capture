#include <stdio.h>
#include "cell.h"

void print_map(struct game_map* game){
  for(int i = 0; i < game -> height; i++) {
    for(int j = 0; j < game -> width; j++) {
      printf("%d ", game -> cells[i][j].is_alive);
    }
    printf("\n");
  }
  printf("\n");
}
