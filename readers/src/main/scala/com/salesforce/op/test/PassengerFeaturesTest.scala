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


trait PassengerFeaturesTest {

  val age = FeatureBuilder.Real[Passenger].extract(_.getAge.toReal).aggregate(MaxReal).asPredictor
  val gender = FeatureBuilder.MultiPickList[Passenger].extract(p => Set(p.getGender).toMultiPickList).asPredictor
  val genderPL = FeatureBuilder.PickList[Passenger].extract(p => p.getGender.toPickList).asPredictor

  val height = FeatureBuilder.RealNN[Passenger]
    .extract(p => Option(p.getHeight).map(_.toDouble).toRealNN(0.0))
    .window(Duration.millis(300))
    .asPredictor

  val heightNoWindow = FeatureBuilder.Real[Passenger].extract(_.getHeight.toReal).asPredictor
  val weight = FeatureBuilder.Real[Passenger].extract(_.getWeight.toReal).asPredictor
  val description = FeatureBuilder.Text[Passenger].extract(_.getDescription.toText).asPredictor
  val boarded = FeatureBuilder.DateList[Passenger].extract(p => Seq(p.getBoarded.toLong).toDateList).asPredictor
  val stringMap = FeatureBuilder.TextMap[Passenger].extract(p => p.getStringMap.toTextMap).asPredictor
  val numericMap = FeatureBuilder.RealMap[Passenger].extract(p => p.getNumericMap.toRealMap).asPredictor
  val booleanMap = FeatureBuilder.BinaryMap[Passenger].extract(p => p.getBooleanMap.toBinaryMap).asPredictor
  val survived = FeatureBuilder.Binary[Passenger].extract(p => Option(p.getSurvived).map(_ == 1).toBinary).asResponse
  val boardedTime = FeatureBuilder.Date[Passenger].extract(_.getBoarded.toLong.toDate).asPredictor
  val boardedTimeAsDateTime = FeatureBuilder.DateTime[Passenger].extract(_.getBoarded.toLong.toDateTime).asPredictor

  val rawFeatures: Array[OPFeature] = Array(
    survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap
  )

}
