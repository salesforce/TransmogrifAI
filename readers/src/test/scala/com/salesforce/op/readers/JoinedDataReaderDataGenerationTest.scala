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

package com.salesforce.op.readers

import com.salesforce.op.aggregators.CutOffTime
import com.salesforce.op.features.types._
import com.salesforce.op.features.{FeatureBuilder, OPFeature}
import com.salesforce.op.test._
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.sql.Row
import org.joda.time.Duration
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}
import org.slf4j.LoggerFactory


@RunWith(classOf[JUnitRunner])
class JoinedDataReaderDataGenerationTest extends FlatSpec with PassengerSparkFixtureTest {

  val log = LoggerFactory.getLogger(this.getClass)

  val newWeight =
    FeatureBuilder.RealNN[PassengerCSV]
      .extract(_.getWeight.toDouble.toRealNN)
      .aggregate(zero = Some(Double.MaxValue), (a, b) => Some(math.min(a.v.getOrElse(0.0), b.v.getOrElse(0.0))))
      .asPredictor

  val newHeight =
    FeatureBuilder.RealNN[PassengerCSV]
      .extract(_.getHeight.toDouble.toRealNN)
      .aggregate(zero = Some(0.0), (a, b) => Some(math.max(a.v.getOrElse(0.0), b.v.getOrElse(0.0))))
      .asPredictor

  val recordTime = FeatureBuilder.DateTime[PassengerCSV].extract(_.getRecordDate.toLong.toDateTime).asPredictor
  val origin = FeatureBuilder.MultiPickList[PassengerProfile].extract(p => Seq(p.getState).toMultiPickList).asPredictor

  Spec[JoinedDataReader[_, _]] should "correctly perform an outer join from two data sources" in {
    val joinedReader = profileReader.outerJoin(dataReader)

    val joinedData = joinedReader.generateDataFrame(Array(survived, age, gender, origin)).collect()

    log.info("Actual data:\n{}", joinedData.mkString("\n"))

    val dataExpected = Array(
      Row(List("NY"), null, 32, List("Female"), "1"),
      Row(List("CO"), null, 33, List("Female"), "2"),
      Row(List("CA"), null, null, List("Male"), "3"),
      Row(null, false, 50, List("Male"), "4"),
      Row(List("NM"), null, 2, List("Female"), "5"),
      Row(List("TX"), true, null, List(), "6"),
      Row(List("UT"), true, null, List(), "6"),
      Row(List("AZ"), null, null, null, "7"))

    log.info("Expected data:\n{}", dataExpected.mkString("\n"))

    joinedData.map(_.get(0)).toSet shouldEqual dataExpected.map(_.get(0)).toSet
    joinedData.map(_.get(1)).toSet shouldEqual dataExpected.map(_.get(1)).toSet
    joinedData.map(_.get(2)).toSet shouldEqual dataExpected.map(_.get(2)).toSet
    joinedData.map(_.get(4)).toSet shouldEqual dataExpected.map(_.get(4)).toSet
  }

  it should "correctly perform an inner join from two data sources" in {
    val joinedReader = profileReader.innerJoin(dataReader)

    val joinedData = joinedReader.generateDataFrame(Array(survived, age, gender, origin)).collect()

    log.info("Actual data:\n{}", joinedData.mkString("\n"))

    val dataExpected = Array(
      Row(List("NY"), null, 32, List("Female"), "1"),
      Row(List("CO"), null, 33, List("Female"), "2"),
      Row(List("CA"), null, null, List("Male"), "3"),
      Row(List("NM"), null, 2, List("Female"), "5"),
      Row(List("TX"), true, null, List(), "6"),
      Row(List("UT"), true, null, List(), "6"))

    log.info("Expected data:\n{}", dataExpected.mkString("\n"))

    joinedData.map(_.get(0)).toSet shouldEqual dataExpected.map(_.get(0)).toSet
    joinedData.map(_.get(1)).toSet shouldEqual dataExpected.map(_.get(1)).toSet
    joinedData.map(_.get(2)).toSet shouldEqual dataExpected.map(_.get(2)).toSet
    joinedData.map(_.get(4)).toSet shouldEqual dataExpected.map(_.get(4)).toSet
  }

  it should "correctly perform a left outer join from two data sources" in {
    val joinedReader = profileReader.leftOuterJoin(dataReader)

    val joinedData = joinedReader.generateDataFrame(Array(survived, age, gender, origin)).collect()

    log.info("Actual data:\n{}", joinedData.mkString("\n"))

    val dataExpected = Array(
      Row(List("NY"), null, 32, List("Female"), "1"),
      Row(List("CO"), null, 33, List("Female"), "2"),
      Row(List("CA"), null, null, List("Male"), "3"),
      Row(List("NM"), null, 2, List("Female"), "5"),
      Row(List("TX"), true, null, List(), "6"),
      Row(List("UT"), true, null, List(), "6"),
      Row(List("AZ"), null, null, null, "7"))

    log.info("Expected data:\n{}", dataExpected.mkString("\n"))

    joinedData.map(_.get(0)).toSet shouldEqual dataExpected.map(_.get(0)).toSet
    joinedData.map(_.get(1)).toSet shouldEqual dataExpected.map(_.get(1)).toSet
    joinedData.map(_.get(2)).toSet shouldEqual dataExpected.map(_.get(2)).toSet
    joinedData.map(_.get(4)).toSet shouldEqual dataExpected.map(_.get(4)).toSet
  }

  it should "correctly join data from three data sources" in {

    val sparkReader = DataReaders.Aggregate.csv[SparkExample](
      path = Some("../test-data/SparkExample.csv"),
      schema = SparkExample.getClassSchema.toString,
      key = _.getLabel.toString,
      aggregateParams = AggregateParams(None, CutOffTime.NoCutoff())
    )

    val stuff = FeatureBuilder.Text[SparkExample].extract(p => Option(p.getStuff).toText).asPredictor
    val joinedReader = profileReader.innerJoin(dataReader).leftOuterJoin(sparkReader)
    val inputFeatures = Array(survived, age, gender, origin, stuff)
    val joinedDataFrame = joinedReader.generateDataFrame(inputFeatures.asInstanceOf[Array[OPFeature]])

    joinedDataFrame.schema.fields.map(_.name).toSet should contain theSameElementsAs inputFeatures.map(_.name) :+
      DataFrameFieldNames.KeyFieldName

    val joinedData = joinedDataFrame.collect()

    log.info("Actual data:\n{}", joinedData.mkString("\n"))

    val dataExpected = Array(
      Row(List("NY"), null, 32, List("Female"), "Logistic regression models are neat", "1"),
      Row(List("CO"), null, 33, List("Female"), null, "2"),
      Row(List("CA"), null, null, List("Male"), null, "3"),
      Row(List("NM"), null, 2, List("Female"), null, "5"),
      Row(List("TX"), true, null, List(), null, "6"),
      Row(List("UT"), true, null, List(), null, "6"))

    log.info("Expected data:\n{}", dataExpected.mkString("\n"))

    joinedData.map(_.get(0)).toSet shouldEqual dataExpected.map(_.get(0)).toSet
    joinedData.map(_.get(1)).toSet shouldEqual dataExpected.map(_.get(1)).toSet
    joinedData.map(_.get(2)).toSet shouldEqual dataExpected.map(_.get(2)).toSet
    joinedData.map(_.get(4)).toSet shouldEqual dataExpected.map(_.get(4)).toSet
    joinedData.map(_.get(5)).toSet shouldEqual dataExpected.map(_.get(5)).toSet
  }

  it should "allow you to join two readers that have the same datatype if you alias the types to be different" in {
    type NewPassenger = Passenger
    val aliasedReader = DataReaders.Simple.avro[NewPassenger](
      path = Some(passengerAvroPath),
      key = _.getPassengerId.toString
    )
    val newDescription = FeatureBuilder.Text[NewPassenger].extract(_.getDescription.toText).asPredictor
    val newBoarded = FeatureBuilder.DateList[NewPassenger].extract(p => Seq(p.getBoarded.toLong).toDateList).asPredictor

    val joinedReader = aliasedReader.innerJoin(dataReader)
    val inputFeatures: Array[OPFeature] = Array(survived, age, boardedTime, newDescription, newBoarded)
    val aggregatedData = joinedReader.generateDataFrame(inputFeatures)

    if (log.isInfoEnabled) aggregatedData.show(false)

    aggregatedData.count() shouldBe 8
    aggregatedData.schema.fields.map(_.name).toSet shouldEqual Set(DataFrameFieldNames.KeyFieldName, survived.name,
      age.name, boardedTime.name, newDescription.name, newBoarded.name)
  }

  it should "perform a secondary aggregation of joined data with using a dummy aggregator" in {
    val sparkReader = DataReaders.Simple.csv[SparkExampleJoin](
      path = Some("../test-data/SparkExampleJoin.csv"),
      schema = SparkExampleJoin.getClassSchema.toString(),
      key = _.getId
    )
    val description = FeatureBuilder.Text[SparkExampleJoin]
      .extract(_.getDescription.toText).asPredictor
    val time = FeatureBuilder.Date[SparkExampleJoin]
      .extract(_.getTimestamp.toLong.toDate).asPredictor

    val secondReader = DataReaders.Simple.csv[JoinTestData](
      path = Some("../test-data/JoinTestData.csv"),
      schema = JoinTestData.getClassSchema.toString(),
      key = _.getId.toString
    )
    val descriptionJoin = FeatureBuilder.Text[JoinTestData].extract(_.getDescription.toText).asPredictor
    val timeJoin = FeatureBuilder.Date[JoinTestData]
      .extract(_.getTimestamp.toDate).asPredictor
    val keyJoin = FeatureBuilder.Text[JoinTestData].extract(_.getSparkId.toText).asPredictor

    val inputFeatures: Array[OPFeature] = Array(description, time, descriptionJoin, timeJoin, keyJoin)

    val joinKeys = JoinKeys(leftKey = DataFrameFieldNames.KeyFieldName,
      rightKey = keyJoin.name,
      resultKey = DataFrameFieldNames.KeyFieldName)

    val timeFilter = TimeBasedFilter(
      condition = new TimeColumn(timeJoin),
      primary = new TimeColumn(time),
      timeWindow = Duration.standardDays(1000)
    )
    val joinedData = sparkReader.outerJoin(secondReader, joinKeys).generateDataFrame(inputFeatures).persist()

    if (log.isInfoEnabled) joinedData.show(false)

    val joinedReader = sparkReader.outerJoin(secondReader, joinKeys).withSecondaryAggregation(timeFilter)
    val aggregatedData = joinedReader.generateDataFrame(inputFeatures).persist()

    if (log.isInfoEnabled) aggregatedData.show(false)

    // right fields unchanged by agg
    joinedData.select(description, time).collect.toSet shouldEqual
      aggregatedData.select(description, time).collect.toSet

    // key 'c' had no aggregation and passes agg filter
    joinedData.filter(r => r.getAs[String](DataFrameFieldNames.KeyFieldName) == "c").collect.head shouldEqual
      aggregatedData.filter(r => r.getAs[String](DataFrameFieldNames.KeyFieldName) == "c").collect.head

    // key 'a' does not pass aggregation filter
    aggregatedData.filter(r => r.getAs[String](DataFrameFieldNames.KeyFieldName) == "a")
      .select(descriptionJoin, timeJoin).collect.head.toSeq shouldEqual Seq(null, null)

    // key 'b' is aggregated
    aggregatedData.filter(r => r.getAs[String](DataFrameFieldNames.KeyFieldName) == "b")
      .select(descriptionJoin, timeJoin).collect.head.toSeq shouldEqual
      Seq("Important too But I hate to write them", 1499175176)
  }

  it should "perform a secondary aggregation of joined data when specified" in {
    val timeFilter = TimeBasedFilter(
      condition = new TimeColumn(boardedTime),
      primary = new TimeColumn(recordTime),
      timeWindow = Duration.standardDays(1000)
    )
    val joinedReader = simpleCsvReader.leftOuterJoin(dataReader)

    val inputFeatures: Array[OPFeature] = Array(
      survived, age, gender, description, stringMap, boarded, height, boardedTime,
      newHeight, newWeight, recordTime
    )

    log.info("Joined & aggregated data:")
    if (log.isInfoEnabled) {
      val nonAgg = joinedReader.generateDataFrame(inputFeatures)
      nonAgg.show(false)
    }

    log.info("After secondary aggregation:")
    val aggregatedData = joinedReader.withSecondaryAggregation(timeFilter).generateDataFrame(inputFeatures).persist()
    if (log.isInfoEnabled) aggregatedData.show(false)

    aggregatedData.select(DataFrameFieldNames.KeyFieldName).collect().map(_.getAs[String](0)).sorted should
      contain theSameElementsAs Array("1", "2", "3", "4", "5", "6")

    aggregatedData.collect(survived) should contain theSameElementsAs
      Array(Binary.empty, Binary.empty, Binary.empty, Binary.empty, Binary.empty, Binary(true))

    aggregatedData.collect(age) should contain theSameElementsAs
      Array(Real.empty, Real.empty, Real.empty, Real(2.0), Real(33.0), Real(50.0))

    aggregatedData.collect(gender) should contain theSameElementsAs
      Array(MultiPickList.empty, MultiPickList.empty, MultiPickList(Set("Female")), MultiPickList(Set("Female")),
        MultiPickList(Set("Male")), MultiPickList(Set("Male")))

    aggregatedData.collect(description) should contain theSameElementsAs
      Array(Text.empty, Text.empty, Text.empty, Text(""),
        Text("stuff this is a description stuff this is a description stuff this is a description"),
        Text("this is a description"))

    aggregatedData.collect(stringMap) should contain theSameElementsAs
      Array(TextMap.empty, TextMap.empty, TextMap(Map("Female" -> "string")),
        TextMap(Map("Female" -> "string")), TextMap(Map("Male" -> "string")),
        TextMap(Map("Male" -> "string string string string string string")))

    aggregatedData.collect(boarded) should contain theSameElementsAs
      Array(DateList.empty, DateList.empty, DateList(Array(1471046100L)), DateList(Array(1471046400L)),
        DateList(Array(1471046300L, 1471046400L, 1471046300L, 1471046400L, 1471046300L, 1471046400L)),
        DateList(Array(1471046600L)))

    // height has a special integration window so this features tests that things included in other
    // features are excluded here
    aggregatedData.collect(height) should contain theSameElementsAs
      Seq(0.0, 0.0, 0.0, 0.0, 0.0, 186.0).toRealNN

    aggregatedData.collect(boardedTime) should contain theSameElementsAs
      Array(Date.empty, Date.empty, Date(1471046100L), Date(1471046400L), Date(1471046400L), Date(1471046600L))

    aggregatedData.collect(newHeight) should contain theSameElementsAs
      Seq(186.0, 168.0, 0.0, 0.0, 186.0, 172.0).toRealNN

    aggregatedData.collect(newWeight) should contain theSameElementsAs
      Seq(96.0, 67.0, Double.MaxValue, Double.MaxValue, 76.0, 78.0).toRealNN

    aggregatedData.collect(recordTime) should contain theSameElementsAs
      Array(DateTime(None), DateTime(None), DateTime(1471045900L), DateTime(1471046000L),
        DateTime(1471046200L), DateTime(1471046400L))
  }

}
