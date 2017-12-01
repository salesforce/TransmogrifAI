/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.test

import com.salesforce.op.features.{FeatureBuilder, OPFeature}
import com.salesforce.op.features.types._
import com.salesforce.op.utils.tuples.RichTuple._
import org.joda.time.Duration


trait PassengerFeaturesTest {

  val age = FeatureBuilder.Real[Passenger]
    .extract(_.getAge.toReal)
    .aggregate((l, r) => (l -> r).map(breeze.linalg.max(_, _)))
    .asPredictor

  val gender = FeatureBuilder.MultiPickList[Passenger].extract(p => Set(p.getGender).toMultiPickList).asPredictor
  val genderPL = FeatureBuilder.PickList[Passenger].extract(p => p.getGender.toPickList).asPredictor

  val height = FeatureBuilder.RealNN[Passenger]
    .extract(_.getHeight.toReal.toRealNN())
    .window(Duration.millis(300))
    .asPredictor

  val heightNoWindow = FeatureBuilder.Real[Passenger]
    .extract(_.getHeight.toReal)
    .asPredictor

  val weight = FeatureBuilder.Real[Passenger].extract(_.getWeight.toReal).asPredictor
  val description = FeatureBuilder.Text[Passenger].extract(_.getDescription.toText).asPredictor
  val boarded = FeatureBuilder.DateList[Passenger].extract(p => Seq(p.getBoarded.toLong).toDateList).asPredictor
  val stringMap = FeatureBuilder.TextMap[Passenger].extract(p => p.getStringMap.toTextMap).asPredictor
  val numericMap = FeatureBuilder.RealMap[Passenger].extract(p => p.getNumericMap.toRealMap).asPredictor
  val booleanMap = FeatureBuilder.BinaryMap[Passenger].extract(p => p.getBooleanMap.toBinaryMap).asPredictor
  val survived = FeatureBuilder.Binary[Passenger].extract(p => Option(p.getSurvived).map(_ == 1).toBinary).asResponse
  val boardedTime = FeatureBuilder.Date[Passenger].extract(_.getBoarded.toLong.toDate).asPredictor

  val rawFeatures: Array[OPFeature] = Array(
    survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)

}
