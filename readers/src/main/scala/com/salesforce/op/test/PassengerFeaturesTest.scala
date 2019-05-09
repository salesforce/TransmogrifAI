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

package com.salesforce.op.test

import com.salesforce.op.features.types._
import com.salesforce.op.features.{FeatureBuilder, OPFeature}
import com.salesforce.op.utils.tuples.RichTuple._
import org.joda.time.Duration
import PassengerFeaturesTestLambdas._
import com.salesforce.op.aggregators.CustomMonoidAggregator

trait PassengerFeaturesTest {

  val age = FeatureBuilder.Real[Passenger].extract(ageFn).aggregate(TestMonoidAggregator).asPredictor
  val gender = FeatureBuilder.MultiPickList[Passenger].extract(genderFn).asPredictor
  val genderPL = FeatureBuilder.PickList[Passenger].extract(genderPLFn).asPredictor
  val height = FeatureBuilder.RealNN[Passenger].extract(heightFn).window(Duration.millis(300)).asPredictor
  val heightNoWindow = FeatureBuilder.Real[Passenger].extract(heightToReal).asPredictor
  val weight = FeatureBuilder.Real[Passenger].extract(weightToReal).asPredictor
  val description = FeatureBuilder.Text[Passenger].extract(descriptionFn).asPredictor
  val boarded = FeatureBuilder.DateList[Passenger].extract(boardedToDL).asPredictor
  val stringMap = FeatureBuilder.TextMap[Passenger].extract(stringMapFn).asPredictor
  val numericMap = FeatureBuilder.RealMap[Passenger].extract(numericMapFn).asPredictor
  val booleanMap = FeatureBuilder.BinaryMap[Passenger].extract(booleanMapFn).asPredictor
  val survived = FeatureBuilder.Binary[Passenger].extract(survivedFn).asResponse
  val boardedTime = FeatureBuilder.Date[Passenger].extract(boardedTimeFn).asPredictor
  val boardedTimeAsDateTime = FeatureBuilder.DateTime[Passenger].extract(boardedDTFn).asPredictor

  val rawFeatures: Array[OPFeature] = Array(
    survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap
  )

}

object TestMonoidAggregator
  extends CustomMonoidAggregator[Real](None, (l, r) => (l -> r).map(breeze.linalg.max(_, _)))
    with Serializable

object PassengerFeaturesTestLambdas {
  def genderFn: Passenger => MultiPickList = p => Set(p.getGender).toMultiPickList
  def genderPLFn: Passenger => PickList = p => p.getGender.toPickList
  def heightFn: Passenger => RealNN = p => Option(p.getHeight).map(_.toDouble).toRealNN(0.0)
  def heightToReal: Passenger => Real = _.getHeight.toReal
  def weightToReal: Passenger => Real = _.getWeight.toReal
  def descriptionFn: Passenger => Text = _.getDescription.toText
  def boardedToDL: Passenger => DateList = p => Seq(p.getBoarded.toLong).toDateList
  def stringMapFn: Passenger => TextMap = p => p.getStringMap.toTextMap
  def numericMapFn: Passenger => RealMap = p => p.getNumericMap.toRealMap
  def booleanMapFn: Passenger => BinaryMap = p => p.getBooleanMap.toBinaryMap
  def survivedFn: Passenger => Binary = p => Option(p.getSurvived).map(_ == 1).toBinary
  def boardedTimeFn: Passenger => Date = _.getBoarded.toLong.toDate
  def boardedDTFn: Passenger => DateTime = _.getBoarded.toLong.toDateTime
  def ageFn: Passenger => Real = _.getAge.toReal
}
