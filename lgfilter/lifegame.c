#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include "lifegame.h"
#include "cell.h"
#include <string.h>

struct cell* get_cell(struct game_map* game, int x, int y) {
  if(x < 0 || y < 0 || x >= game -> width || y >= game -> height) {
    return NULL;
  }
  return &(game -> cells[y][x]);
}

void* summation_x(void* arg) {
  struct summation_args* args = (struct summation_args*) arg;
  args -> life_map[args -> y][0] = 0;
  args -> life_map[args -> y][1] = 0;
  int cell_y = args -> y - 2;
  args -> life_map[args -> y][2] =
    args -> game -> cells[cell_y][0].is_alive;
  for(int j = 3; j < args -> game -> width + 2; j++) {
    args -> life_map[args -> y][j] = 
      args -> life_map[args -> y][j - 1] + 
      args -> game -> cells[cell_y][j - 2].is_alive;
  }
  return 0;
}

void* summation_y(void* arg) {
  struct summation_args* args = (struct summation_args*) arg;
  for(int j = 3; j < args -> game -> height + 2; j++) {
    args -> life_map[j][args -> x] +=
      args -> life_map[j - 1][args -> x];
  }
  args -> life_map[args -> game -> height + 2][args -> x] = 
    args -> life_map[args -> game -> height + 1][args -> x];
  return 0;
}

int** cumlative_sum(struct game_map* game) {
  int** life_map = (int**)malloc(sizeof(int*) * (game -> height + 3));
  unsigned long width = sizeof(int) * (game -> width + 3);
  for(int i = 0; i < game -> height + 3; i++) {
    life_map[i] = (int*)malloc(width);
  }
  memset(life_map[0], 0, width);
  memset(life_map[1], 0, width);

  pthread_t xthreads[game -> height];
  struct summation_args args[
    game -> height > game -> width ?
      game -> height : game -> width
  ];

  for(int i = 0; i < game -> height; i++) {
    args[i].life_map = life_map;
    args[i].game = game;
    args[i].y = i + 2;
    pthread_create(&xthreads[i], NULL, summation_x, &args[i]);
  }

  for(int i = 0; i < game -> height; i++) {
    pthread_join(xthreads[i], NULL);
  }

  pthread_t ythreads[game -> width];
  for(int i = 0; i < game -> width; i++) {
    args[i].life_map = life_map;
    args[i].game = game;
    args[i].x = i + 2;
    pthread_create(&ythreads[i], NULL, summation_y, &args[i]);
  }

  for(int i = 0; i < game -> width; i++) {
    pthread_join(ythreads[i], NULL);
  }

  int x = game -> width + 2;
  for(int i = 2; i < game -> height + 3; i++) {
    life_map[i][x] = life_map[i][game -> width];
  }
  
  return life_map;
}

void close_cumlative_sum(struct game_map* game, int** cumsum) {
  for(int i = 0; i < game -> height + 3; i++) {
    free(cumsum[i]);
  }
  free(cumsum);
}

void spawn(struct game_map* game, int x, int y) {
  game -> cells[y][x].is_alive = true;

  int r = 0, g = 0, b = 0, count = 0;
  for(int i = y - 1; i <= y + 1; i++) {
    for(int j = x - 1; j <= x + 1; j++) {
      struct cell* target = get_cell(game, j, i);
      if(target != NULL && target -> is_alive) {
        r += target -> r;
        g += target -> g;
        b += target -> b;
        count++;
      }
    }
  }

  game -> cells[y][x].r = r / count;
  game -> cells[y][x].g = g / count;
  game -> cells[y][x].b = b / count;
}

void kill(struct game_map* game, int x, int y) {
  game -> cells[y][x].is_alive = false;
}

void next(struct game_map* game) {
  int** cs = cumlative_sum(game);
  for(int i = 0; i < game -> height; i++) {
    for(int j = 0; j < game -> width; j++) {
      int lifes = 
          cs[i + 3][j + 3]
          + cs[i][j]
          - cs[i + 3][j]
          - cs[i][j + 3];
      if(lifes == 2) {
        kill(game, j, i);
      } else if(lifes == 3) {
        spawn(game, j, i);
      } else if(lifes > 4) {
        kill(game, j, i);
      }
    }
  }
  close_cumlative_sum(game, cs);
}
