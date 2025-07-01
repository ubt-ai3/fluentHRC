#include <iostream>

#include "csvworker.h"

int main() {
	std::string path("W:\\DB_Forschung\\FlexCobot\\11.Unterprojekte\\DS_Nico_Höllerich\\05.Rohdaten\\MMK.Bausteine\\Motion Tracking.no_backup\\Team.0.IDs.-1.-2.Trial.1.csv");

	CsvWorker csv;
	csv.loadFromFile(path);

	std::cout << csv.getRowsCount() << "  " << csv.getColumnsCount() << std::endl;
}