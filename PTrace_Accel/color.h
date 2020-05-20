#pragma once
#include <stdint.h>

class color{
public:
	color(uint32_t e) {
		this->e = e;
	}

	color(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
		e = r | (g << 8) | (b << 16) | (a << 24);
	}


	uint8_t getR() { return (e & 0xFF000000) >> 24; }
	uint8_t getG() { return (e & 0x00FF0000) >> 16; }
	uint8_t getB() { return (e & 0x0000FF00) >> 8; }
	uint8_t getA() { return  e & 0x000000FF; }

	void setR(uint8_t r) {
		e &= 0x00FFFFFF;
		e |= r;
	}
	void setG(uint8_t g) {
		e &= 0xFF00FFFF;
		e |= g << 8;
	}
	void setB(uint8_t b) {
		e &= 0xFFFF00FF;
		e |= b << 16;
	}
	void setA(uint8_t a) {
		e &= 0xFFFFFF00;
		e |= a << 24;
	}

	uint32_t getColor() { return e; }


private:
	uint32_t e;
};