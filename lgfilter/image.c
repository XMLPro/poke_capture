#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "image.h"

void load_bmp(struct bmp_image* bmp, char* filename) {
  FILE* file = fopen(filename, "rb");
  if(file == NULL) {
    puts("file not found.");
  }

  bmp -> header = (unsigned char*)malloc(sizeof(unsigned char) * HEADER_SIZE);
  fread(bmp -> header, sizeof(unsigned char), HEADER_SIZE, file);

  unsigned int width, height;
  memcpy(&width, bmp -> header + 18, sizeof(unsigned int));
  memcpy(&height, bmp -> header + 22, sizeof(unsigned int));

  unsigned int true_width = width + (4 - width % 4) % 4;
  bmp -> width = width;
  bmp -> height = height;
  bmp -> true_width = true_width;

  unsigned char data[height * true_width * 3];
  fread(data, sizeof(unsigned char), height * true_width * 3, file);
  fclose(file);

  unsigned char*** image_data = (unsigned char***)malloc(sizeof(unsigned char**) * height);
  for(unsigned int i = 0; i < height; i++) {
    image_data[i] = (unsigned char**)malloc(sizeof(unsigned char*) * width);
    for(unsigned int j = 0; j < width; j++) {
      image_data[i][j] = (unsigned char*)malloc(sizeof(unsigned char) * 3);
      for(char rgb = 0; rgb < 3; rgb++) {
        image_data[i][j][rgb] = data[i*width*3 + j*3 + rgb];
      }
    }
  }

  bmp -> data = image_data;
}

void write_bmp(struct bmp_image* bmp, char* filename) {
  FILE* write_target = fopen(filename, "wb");
  unsigned char data[bmp -> height * bmp -> true_width * 3];
  int diff = bmp -> true_width - bmp -> width;
  for(unsigned int i = 0; i < bmp -> height; i++) {
    for(unsigned int j = 0; j < bmp -> width; j++) {
      for(char rgb = 0; rgb < 3; rgb++) {
        data[i*bmp -> width*3 + j*3 + rgb] = bmp -> data[i][j][rgb];
      }
    }
    for(int d = 0; d < diff; d++) {
      data[i*bmp -> width*3 + bmp -> width*3 + d] = 0;
    }
  }
  fwrite(bmp -> header, sizeof(unsigned char), HEADER_SIZE, write_target);
  fwrite(data, sizeof(unsigned char), bmp -> height * bmp -> true_width * 3, write_target);
  fclose(write_target);
}

void close_bmp(struct bmp_image* bmp) {
  free(bmp -> header);

  for(unsigned int i = 0; i < bmp -> height; i++) {
    for(unsigned int j = 0; j < bmp -> width; j++) {
      free(bmp -> data[i][j]);
    }
    free(bmp -> data[i]);
  }
  free(bmp -> data);
}
