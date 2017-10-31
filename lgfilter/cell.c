#include <stdlib.h>
#include "cell.h"

void init_game_map(struct game_map* game, int width, int height) {
  struct cell** cells = (struct cell**)malloc(sizeof(struct cell*) * height);
  for(int i = 0; i < height; i++) {
    cells[i] = (struct cell*)malloc(sizeof(struct cell) * width);
    for(int j = 0; j < width; j++) {
      cells[i][j].x = j;
      cells[i][j].y = i;
    }
  }

  game -> width = width;
  game -> height = height;
  game -> cells = cells;
}

void close_game_map(struct game_map* game) {
  for(int i = 0; i < game -> height; i++) {
    free(game -> cells[i]);
  }
  free(game -> cells);
}
