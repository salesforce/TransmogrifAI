/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.hw.titanic

import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._

trait TitanicFeatures extends Serializable {

  val survived =
    FeatureBuilder.RealNN[Passenger].extract(_.getSurvived.toDouble.toRealNN).asResponse

  val pClass =
    FeatureBuilder.PickList[Passenger].extract(d => Option(d.getPclass).map(_.toString).toPickList).asPredictor

  val name = FeatureBuilder.Text[Passenger].extract(d => Option(d.getName).toText).asPredictor

  val sex = FeatureBuilder.PickList[Passenger].extract(d => Option(d.getSex).toPickList).asPredictor

  val age = FeatureBuilder.Real[Passenger].extract(d => Option(Double.unbox(d.getAge)).toReal).asPredictor

  val sibSp = FeatureBuilder.PickList[Passenger].extract(d => Option(d.getSibSp).map(_.toString).toPickList).asPredictor

  val parch = FeatureBuilder.PickList[Passenger].extract(d => Option(d.getParch).map(_.toString).toPickList).asPredictor

  val ticket = FeatureBuilder.PickList[Passenger].extract(d => Option(d.getTicket).toPickList).asPredictor

  val fare = FeatureBuilder.Real[Passenger].extract(d => Option(Double.unbox(d.getFare)).toReal).asPredictor

  val cabin = FeatureBuilder.PickList[Passenger].extract(d => Option(d.getCabin).toPickList).asPredictor

  val embarked = FeatureBuilder.PickList[Passenger].extract(d => Option(d.getEmbarked).toPickList).asPredictor

}
