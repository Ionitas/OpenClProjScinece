__kernel void hello(__global char* string)
{
string[0] = 'H';
string[1] = 'e';
string[2] = 'l';
string[3] = 'l';
string[4] = 'o';
string[5] = ',';
string[6] = 'P';
string[7] = 'I';
string[8] = 'D';
string[9] = 'O';
string[10] = 'R';
string[11] = 'm';
string[12] = 'a';
string[13] = 'n';
string[14] = ')';
string[15] = '\0';
}