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

import com.salesforce.op.aggregators.MaxReal
import com.salesforce.op.features.types._
import com.salesforce.op.features.{FeatureBuilder, OPFeature}
import org.joda.time.Duration
import PassengerFeaturesTestLambdas._


trait PassengerFeaturesTest {
  val age = FeatureBuilder.Real[Passenger].extract(ageFn).aggregate(MaxReal).asPredictor
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

object PassengerFeaturesTestLambdas {
  private class GenderFn extends Function1[Passenger, MultiPickList] with Serializable {
    def apply(p: Passenger): MultiPickList = Set(p.getGender).toMultiPickList
  }

  private class GenderPLFn extends Function1[Passenger, PickList] with Serializable {
    def apply(p: Passenger): PickList = p.getGender.toPickList
  }

  private class HeightFn extends Function1[Passenger, RealNN] with Serializable {
    def apply(p: Passenger): RealNN = Option(p.getHeight).map(_.toDouble).toRealNN(0.0)
  }

  private class HeightToReal extends Function1[Passenger, Real] with Serializable {
    def apply(p: Passenger): Real = p.getHeight.toReal
  }

  private class WeightToReal extends Function1[Passenger, Real] with Serializable {
    def apply(p: Passenger): Real = p.getWeight.toReal
  }

  private class DescriptionFn extends Function1[Passenger, Text] with Serializable {
    def apply(p: Passenger): Text = p.getDescription.toText
  }

  private class BoardedToDL extends Function1[Passenger, DateList] with Serializable {
    def apply(p: Passenger): DateList = Seq(p.getBoarded.toLong).toDateList
  }

  private class StringMapFn extends Function1[Passenger, TextMap] with Serializable {
    def apply(p: Passenger): TextMap = p.getStringMap.toTextMap
  }

  private class NumericMapFn extends Function1[Passenger, RealMap] with Serializable {
    def apply(p: Passenger): RealMap = p.getNumericMap.toRealMap
  }

  private class BooleanMapFn extends Function1[Passenger, BinaryMap] with Serializable {
    def apply(p: Passenger): BinaryMap = p.getBooleanMap.toBinaryMap
  }

  private class SurvivedFn extends Function1[Passenger, Binary] with Serializable {
    def apply(p: Passenger): Binary = Option(p.getSurvived).map(_ == 1).toBinary
  }

  private class BoardedTimeFn extends Function1[Passenger, Date] with Serializable {
    def apply(p: Passenger): Date = p.getBoarded.toLong.toDate
  }

  private class BoardedDTFn extends Function1[Passenger, DateTime] with Serializable {
    def apply(p: Passenger): DateTime = p.getBoarded.toLong.toDateTime
  }

  private class AgeFn extends Function1[Passenger, Real] with Serializable {
    def apply(p: Passenger): Real = p.getAge.toReal
  }

  def genderFn: Passenger => MultiPickList = new GenderFn
  def genderPLFn: Passenger => PickList = new GenderPLFn
  def heightFn: Passenger => RealNN = new HeightFn
  def heightToReal: Passenger => Real = new HeightToReal
  def weightToReal: Passenger => Real = new WeightToReal
  def descriptionFn: Passenger => Text = new DescriptionFn
  def boardedToDL: Passenger => DateList = new BoardedToDL
  def stringMapFn: Passenger => TextMap = new StringMapFn
  def numericMapFn: Passenger => RealMap = new NumericMapFn
  def booleanMapFn: Passenger => BinaryMap = new BooleanMapFn
  def survivedFn: Passenger => Binary = new SurvivedFn
  def boardedTimeFn: Passenger => Date = new BoardedTimeFn
  def boardedDTFn: Passenger => DateTime = new BoardedDTFn
  def ageFn: Passenger => Real = new AgeFn
}
