#include <iostream>
#include <algorithm>
#include "csvworker.h"


// If counts of rows and columns !=0, this fills vector of rows "data";
// each of rows has cols_count empty strings
CsvWorker::CsvWorker(char delim, unsigned int rows_count, unsigned int cols_count)
{
	this->delimiter  = delim;
    this->rows_count = rows_count;
    this->cols_count = cols_count;
    row rw(cols_count, "");
	data.resize(rows_count, rw);
}

// This loads .csv file and fills vector of rows "data";
// counts of rows and columns determines from file data.
void CsvWorker::loadFromFile(std::string filename)
{
    // Load file data
	std::deque<std::string> readed_data;
	loadFile(filename, readed_data);
	// Parse readed data
	parseData(readed_data);
}

void CsvWorker::parseData(std::deque<std::string> &readed_data)
{
	char cur_field[max_field_size];
	int cur_field_size = 0;
	int cur_col = 0;
	char c;

	data.emplace_back();
	row *rw = new row(rows_count);
	for (auto str : readed_data) {
		int sz = str.size();
		const char * cdata = str.data();
		for (int i = 0; i < sz; ++i) {
			c = cdata[i];
			if (c != delimiter && (int)c != 10 && (int)c != 13) {
				cur_field[cur_field_size] = c;
				cur_field_size++;
			}
			else if (c == delimiter) {
				// new field
				rw->emplace_back(cur_field, cur_field_size);
				cur_field_size = 0;
			} else {
				// handle CRLF
				if (i < sz && ((int)(cdata[i + 1]) == 10 || (int)(cdata[i + 1]) == 13))
					i++;
				// new row
					if (!rw->empty()) {
						data.emplace_back();
						data[cur_col] = std::move(*rw);
						cur_col++;
						delete rw;
						rw = new row(rows_count);
					}
			}
		}
	}
	delete rw;
	data.pop_back();
	rows_count = data.size();
	if(rows_count>0)
		for(const row& r: data)
			cols_count = std::max(cols_count, (unsigned int) data[0].size());
}


// This loads data from file to rd
void CsvWorker::loadFile(std::string filename, std::deque<std::string> &readed_data)
{
	// File to read
	FILE *fin;
	char buffer[BLOCK_SIZE];
	int ok = fopen_s(&fin, filename.c_str(), "rb");
	if (!ok) {
		// The end of the file
		_fseeki64(fin, 0, SEEK_END);
		// Get size of the file
		long long m_file_size = _ftelli64(fin);
		// Go to start
		_fseeki64(fin, 0, SEEK_SET);
		// Read and write by "BLOCK_SIZE" bytes
		int blocks_count = m_file_size / BLOCK_SIZE;
		for (size_t i = 0; i <= blocks_count; ++i) {
			int readed = fread(buffer, sizeof(unsigned char), BLOCK_SIZE, fin);
			readed_data.emplace_back((char*)buffer, readed);
		}
		fclose(fin);
	}
	else
	{
		std::cout << "Can't open file \"" << filename << '\"' << std::endl;
		throw;
	}
}


// This saves in .csv file vector of rows "this->data".
// This vector must be already filled.
void CsvWorker::saveToFile(std::string filename)
{
    if(this->cols_count == 0) // if count of columns equals 0, generate error
    {
		std::cout << "Wrong csv format: columns=0" << std::endl;
        throw;
    }
    if(this->rows_count == 0) // if count of rows equals 0, generate error
    {
		std::cout << "Wrong csv format: rows=0" << std::endl;
        throw;
    }
    // Try to open file
	std::ofstream out(filename.c_str());
    if( !out )
    {
		std::cout << "Can't create or open file \"" << filename << '\"' << std::endl;
        throw;
    }
    // write "data" in file
    for(unsigned int i = 0; i < this->rows_count; ++i)
    {
        row & rw = data[i]; // create reference to part of data 
        unsigned int delimiters_count = this->cols_count - 1; // count of delimiters
        // write row in file
        for(unsigned int j = 0; j < this->cols_count; ++j)
        {
            out << rw[j];
            if(j < delimiters_count)
                out << delimiter;
        }
        // if row is not last, write endline
        if(i < this->rows_count - 1)
            out << std::endl;
    }
    out.close(); // close file
}

// This gets reference to row ( typedef std::vector<std::string> row; )
// in data ( std::vector<row> data; ) with row index row_n.
row &CsvWorker::getRowRef(unsigned int row_n)
{
    if(row_n >= this->rows_count) // if row index is wrong, generate error
    {
		std::cout << "Wrong row index: " << row_n << ", but max index is: " << this->rows_count - 1 << std::endl;
        throw;
    }
    return data[row_n];
}

// This gets reference to row's field ( typedef std::vector<std::string> row; )
// with row index row_n and field(column) index col_n.
const std::string* CsvWorker::getFieldOrNull(unsigned int row_n, unsigned int col_n)
{
	row& rw = getRowRef(row_n);
	if (col_n >= rw.size()) // if field index is wrong, generate error
	{
		return nullptr;
	}
	return &rw[col_n];
}

// This gets reference to row's field ( typedef std::vector<std::string> row; )
// with row index row_n and field(column) index col_n.
std::string & CsvWorker::getFieldRef(unsigned int row_n, unsigned int col_n)
{
    row &rw = getRowRef(row_n);
    if(col_n >= rw.size()) // if field index is wrong, generate error
    {
		std::cout << "Wrong field index: " << col_n << ", but max index is: " << this->cols_count - 1 << std::endl;
        throw;
    }
    return rw[col_n];
}

// This gets row's field ( typedef std::vector<std::string> row; )
// with row index row_n and field(column) index col_n.
std::string CsvWorker::getField(unsigned int row_n, unsigned int col_n)
{
    return getFieldRef(row_n, col_n);
}