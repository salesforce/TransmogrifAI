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

import com.salesforce.op.aggregators.{MaxInteger, MaxReal}
import com.salesforce.op.features.types._
import com.salesforce.op.features.{FeatureBuilder, OPFeature}
import org.joda.time.Duration
import PassengerFeaturesTest._


trait PassengerFeaturesTest {
  val age = FeatureBuilder.Integer[Passenger].extract(new AgeExtract).aggregate(MaxInteger).asPredictor
  val gender = FeatureBuilder.MultiPickList[Passenger].extract(new GenderAsMultiPickListExtract).asPredictor
  val genderPL = FeatureBuilder.PickList[Passenger].extract(new GenderAsPickListExtract).asPredictor
  val height = FeatureBuilder.RealNN[Passenger].extract(new HeightToRealNNExtract)
    .window(Duration.millis(300)).asPredictor
  val heightNoWindow = FeatureBuilder.Real[Passenger].extract(new HeightToRealExtract).asPredictor
  val weight = FeatureBuilder.Real[Passenger].extract(new WeightToRealExtract).asPredictor
  val description = FeatureBuilder.Text[Passenger].extract(new DescriptionExtract).asPredictor
  val boarded = FeatureBuilder.DateList[Passenger].extract(new BoardedToDateListExtract).asPredictor
  val stringMap = FeatureBuilder.TextMap[Passenger].extract(new StringMapExtract).asPredictor
  val numericMap = FeatureBuilder.RealMap[Passenger].extract(new NumericMapExtract).asPredictor
  val booleanMap = FeatureBuilder.BinaryMap[Passenger].extract(new BooleanMapExtract).asPredictor
  val survived = FeatureBuilder.Binary[Passenger].extract(new SurvivedExtract).asResponse
  val boardedTime = FeatureBuilder.Date[Passenger].extract(new BoardedToDateExtract).asPredictor
  val boardedTimeAsDateTime = FeatureBuilder.DateTime[Passenger].extract(new BoardedToDateTimeExtract).asPredictor

  val rawFeatures: Array[OPFeature] = Array(
    survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap
  )

}

object PassengerFeaturesTest {

  class GenderAsMultiPickListExtract extends Function1[Passenger, MultiPickList] with Serializable {
    def apply(p: Passenger): MultiPickList = Set(p.getGender).toMultiPickList
  }
  class GenderAsPickListExtract extends Function1[Passenger, PickList] with Serializable {
    def apply(p: Passenger): PickList = p.getGender.toPickList
  }
  class HeightToRealNNExtract extends Function1[Passenger, RealNN] with Serializable {
    def apply(p: Passenger): RealNN = Option(p.getHeight).map(_.toDouble).toRealNN(0.0)
  }
  class HeightToRealExtract extends Function1[Passenger, Real] with Serializable {
    def apply(p: Passenger): Real = p.getHeight.toReal
  }
  class WeightToRealExtract extends Function1[Passenger, Real] with Serializable {
    def apply(p: Passenger): Real = p.getWeight.toReal
  }
  class DescriptionExtract extends Function1[Passenger, Text] with Serializable {
    def apply(p: Passenger): Text = p.getDescription.toText
  }
  class BoardedToDateListExtract extends Function1[Passenger, DateList] with Serializable {
    def apply(p: Passenger): DateList = Seq(p.getBoarded.toLong).toDateList
  }
  class BoardedToDateExtract extends Function1[Passenger, Date] with Serializable {
    def apply(p: Passenger): Date = p.getBoarded.toLong.toDate
  }
  class BoardedToDateTimeExtract extends Function1[Passenger, DateTime] with Serializable {
    def apply(p: Passenger): DateTime = p.getBoarded.toLong.toDateTime
  }
  class SurvivedExtract extends Function1[Passenger, Binary] with Serializable {
    def apply(p: Passenger): Binary = Option(p.getSurvived).map(_ == 1).toBinary
  }
  class StringMapExtract extends Function1[Passenger, TextMap] with Serializable {
    def apply(p: Passenger): TextMap = p.getStringMap.toTextMap
  }
  class NumericMapExtract extends Function1[Passenger, RealMap] with Serializable {
    def apply(p: Passenger): RealMap = p.getNumericMap.toRealMap
  }
  class BooleanMapExtract extends Function1[Passenger, BinaryMap] with Serializable {
    def apply(p: Passenger): BinaryMap = p.getBooleanMap.toBinaryMap
  }
  class AgeExtract extends Function1[Passenger, Integer] with Serializable {
    def apply(p: Passenger): Integer = p.getAge.toInteger
  }

}
