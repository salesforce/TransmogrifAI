/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.readers

import com.salesforce.op.aggregators.CutOffTime
import com.salesforce.op.features.types._
import com.salesforce.op.test._
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.sql.Row
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class DataGenerationTest extends FlatSpec with PassengerSparkFixtureTest {

  private def column(data: Array[Row])(i: Int): Array[Any] = data map (_.get(i))

  Spec[DataReader[_]] should "read data and extract the features to create a dataframe" in {
    val simpleAvroReader = DataReaders.Simple.avro[Passenger](
      path = Some(passengerAvroPath),
      key = _.getPassengerId.toString
    )
    val dataSet = simpleAvroReader.generateDataFrame(rawFeatures).collect()

    val dataExpected = Array(
      Row("1", false, 32, Array("Female"), 168, 67, null, Array[Long](1471046200),
        Map("Female" -> "string"), Map("Female" -> 1.0), Map("Female" -> false)),
      Row("2", true, 33, Array("Female"), 172, 78, "", Array[Long](1471046400),
        Map("Female" -> "string"), Map("Female" -> 1.0), Map("Female" -> false)),
      Row("3", true, null, Array("Male"), 186, 96, "this is a description", Array[Long](1471046600),
        Map("Male" -> "string"), Map("Male" -> 1.0), Map("Male" -> false)),
      Row("4", false, 10, Array("Male"), 177, 76, "stuff", Array[Long](1471046400),
        Map("Male" -> "string"), Map("Male" -> 1.0), Map("Male" -> false)),
      Row("5", false, 2, Array("Female"), 168, 67, null, Array[Long](1471046100),
        Map("Female" -> "string"), Map("Female" -> 1.0), Map("Female" -> false)),
      Row("6", true, 40, Array("Female"), 172, 78, "", Array[Long](1471046600),
        Map("Female" -> "string"), Map("Female" -> 1.0), Map("Female" -> false)),
      Row("4", true, 50, Array("Male"), 186, 96, "this is a description", Array[Long](1471046400),
        Map("Male" -> "string"), Map("Male" -> 1.0), Map("Male" -> false)),
      Row("4", false, 19, Array("Male"), 177, 76, "stuff", Array[Long](1471046300),
        Map("Male" -> "string"), Map("Male" -> 1.0), Map("Male" -> false)))

    val actualAt = column(dataSet) _
    val expectedAt = column(dataExpected) _

    actualAt(0).toSet shouldEqual expectedAt(0).toSet
    actualAt(1).toSet shouldEqual expectedAt(1).toSet
    actualAt(2).toSet shouldEqual expectedAt(2).toSet
    dataSet.map(_.getSeq[String](3).head).toSet shouldEqual dataExpected.map(_.getAs[Array[String]](3).head).toSet
    actualAt(4).toSet shouldEqual expectedAt(4).toSet
    actualAt(5).toSet shouldEqual expectedAt(5).toSet
    actualAt(6).toSet shouldEqual expectedAt(6).toSet
    dataSet.map(_.getSeq[Long](7).head).toSet shouldEqual dataExpected.map(_.getAs[Array[Long]](7).head).toSet
    actualAt(8).toSet shouldEqual expectedAt(8).toSet
    actualAt(9).toSet shouldEqual expectedAt(9).toSet
    actualAt(10).toSet shouldEqual expectedAt(10).toSet
  }

  Spec[AggregateDataReader[_]] should "read data, extract and aggregate the features to create a dataframe" in {
    val dataExpected = Array(
      Row("1", null, 32, List("Female"), 168, 67, "", List(1471046200),
        Map("Female" -> "string"), Map("Female" -> 1.0), Map("Female" -> false)),
      Row("2", null, 33, List("Female"), 0.0, 78, null, List(1471046400),
        Map("Female" -> "string"), Map("Female" -> 1.0), Map("Female" -> false)),
      Row("3", null, null, List("Male"), 186, 96, "this is a description", List(1471046600),
        Map("Male" -> "string"), Map("Male" -> 1.0), Map("Male" -> false)),
      Row("4", false, 50, List("Male"), 363, 172, "this is a description stuff", List(1471046400, 1471046300),
        Map("Male" -> "string string"), Map("Male" -> 2.0), Map("Male" -> false)),
      Row("5", null, 2, List("Female"), 0.0, 67, "", List(1471046100),
        Map("Female" -> "string"), Map("Female" -> 1.0), Map("Female" -> false)),
      Row("6", true, null, null, 0.0, null, null, null, null, null, null)
    )
    val passenger4 = passengersArray.filter(_.get(0) == "4").head
    passenger4.getSeq[String](3).head shouldEqual "Male"
    passenger4.getSeq[String](3).length shouldEqual 1
    passenger4.getSeq[Long](7).length shouldEqual 2

    passengersArray.map(_.get(0)).length shouldEqual column(dataExpected)(0).length
    for { i <- 1 to 10 } {
      column(passengersArray)(i).toSet shouldEqual column(dataExpected)(i).toSet
    }
  }


  Spec[ConditionalDataReader[_]] should
    "read data, extract and do conditional aggregation on the features to create a dataframe" in {

    val dataSource = DataReaders.Conditional.avro[Passenger](
      path = Some(passengerAvroPath),
      key = _.getPassengerId.toString,
      conditionalParams = ConditionalParams(
        timeStampFn = _.getRecordDate.toLong, // Record field which defines the date for the rest of the columns
        targetCondition = _.getBoarded >= 1471046600, // Function to figure out if target event has occurred
        responseWindow = None, // How many days after target event to include in response aggregation
        predictorWindow = None, // How many days before target event to include in predictor aggregation
        timeStampToKeep = TimeStampToKeep.Min
      )
    )

    val dataSet = dataSource.generateDataFrame(rawFeatures).collect()

    /* Logic on this should be double checked - conditional extractors
     * take target function if target is true get date from that record and use it
     * for cutoff date between predictors and responses
     * */
    val dataExpected = Array(
      Row("1", null, 32, List("Female"), 0.0, 67, "", List(1471046200),
        Map("Female" -> "string"), Map("Female" -> 1.0), Map("Female" -> false)),
      Row("2", null, 33, List("Female"), 0.0, 78, null, List(1471046400),
        Map("Female" -> "string"), Map("Female" -> 1.0), Map("Female" -> false)),
      Row("3", true, null, null, 0.0, null, "", null, null, null, null),
      Row("5", null, 2, List("Female"), 0.0, 67, "", List(1471046100),
        Map("Female" -> "string"), Map("Female" -> 1.0), Map("Female" -> false)),
      Row("6", true, null, null, 0.0, null, null, null, null, null, null),
      Row("4", null, 50, List("Male"), 0.0, 248, "this is a description stuff stuff",
        List(1471046400, 1471046400, 1471046300), Map("Male" -> "string string string"),
        Map("Male" -> 3.0), Map("Male" -> false))
    )
    val passenger4 = dataSet.filter(_.get(0) == "4").head
    passenger4.getSeq[String](3).head shouldEqual "Male"
    passenger4.getSeq[String](3).length shouldEqual 1
    passenger4.getSeq[Long](7).length shouldEqual 3

    val actualAt = column(dataSet) _
    val expectedAt = column(dataExpected) _

    actualAt(0).length shouldEqual expectedAt(0).length
    for { i <- 1 to 10 } {
      actualAt(i).toSet shouldEqual expectedAt(i).toSet
    }
  }

  it should
    "read data, extract and do conditional aggregation on the features to create a dataframe with a user cutoff fn" in {
    val dataSource = DataReaders.Conditional.avro[Passenger](
      path = Some(passengerAvroPath),
      key = _.getPassengerId.toString,
      conditionalParams = ConditionalParams(
        timeStampFn = _.getRecordDate.toLong,
        targetCondition = _.getBoarded >= 1471046600,
        responseWindow = None,
        predictorWindow = None,
        cutOffTimeFn = Some((key, rows) => CutOffTime.UnixEpoch(0)), // cut everything off
        timeStampToKeep = TimeStampToKeep.Min
      )
    )
    val dataSet = dataSource.generateDataFrame(rawFeatures)
    dataSet.collect(age, gender, weight, description, boarded).foreach(r =>
      r.productIterator.forall(_.asInstanceOf[FeatureType].isEmpty) shouldBe true
    )
    dataSet.collect(stringMap, numericMap, booleanMap).foreach(r =>
      r.productIterator.forall(_.asInstanceOf[FeatureType].isEmpty) shouldBe true
    )
    dataSet.collect(height).foreach(_ shouldBe RealNN(0.0))
  }

}
