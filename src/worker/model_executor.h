/*
Author: KB 
Date: 2026-01-19 
Purpose: Header file for model_executor.cc
File: model_executor.h 
*/

#pragma once //preprocessor directive that asks the compiler to include a header file only once and not repeat it; To prevent duplications..

#include <string>
#include <vector>

#include "process_executor.h"

int ExecuteModel(const ModelSpec& spec,
		 const std::string& input,
		 std::string* ouput
		 );
