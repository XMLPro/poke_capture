#ifndef INCLUDE_IMAGE
#define INCLUDE_IMAGE

#define HEADER_SIZE 54

struct bmp_image {
  unsigned int width, height, true_width;
  unsigned char* header;
  unsigned char*** data;
};

void load_bmp(struct bmp_image*, char* filename);
void write_bmp(struct bmp_image*, char* filename);
void close_bmp(struct bmp_image*);

#endif
