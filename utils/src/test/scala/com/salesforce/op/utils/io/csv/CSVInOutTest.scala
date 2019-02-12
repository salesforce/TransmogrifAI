/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.utils.io.csv

import com.salesforce.op.test.TestSparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{AnalysisException, DataFrame}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class CSVInOutTest extends FlatSpec with TestSparkContext {
  private val csvReader = new CSVInOut(CSVOptions(header = true))
  private val csvFile = s"$testDataDir/PassengerDataAllWithHeader.csv"

  Spec[CSVInOut] should "throw error for bad file paths with DataFrame" in {
    val error = intercept[AnalysisException](csvReader.readDataFrame("/bad/file/path/read/dataframe"))
    error.getMessage should endWith ("Path does not exist: file:/bad/file/path/read/dataframe;")
  }

  it should "throw error for bad file paths with RDD" in {
    val error = intercept[AnalysisException](csvReader.readRDD("/bad/file/path/read/rdd"))
    error.getMessage should endWith ("Path does not exist: file:/bad/file/path/read/rdd;")
  }

  it should "read a CSV file to DataFrame" in {
    val res = csvReader.readDataFrame(csvFile)
    res shouldBe a[DataFrame]
    res.count shouldBe 891
  }

  it should "read a CSV file to RDD" in {
    val res = csvReader.readRDD(csvFile)
    res shouldBe a[RDD[_]]
    res.count shouldBe 891
  }
}
