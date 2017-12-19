/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.hw.titanic

import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._

trait TitanicFeatures extends Serializable {

  val survived = FeatureBuilder.RealNN[Passenger]
    .extract(_.getSurvived.toDouble.toRealNN).asResponse

  val pClass = FeatureBuilder.MultiPickList[Passenger]
    .extract(d => Option(d.getPclass).map(_.toString).toSet[String].toMultiPickList).asPredictor

  val name = FeatureBuilder.Text[Passenger]
    .extract(d => Option(d.getName).toText).asPredictor

  val sex = FeatureBuilder.MultiPickList[Passenger]
    .extract(d => Option(d.getSex).toSet[String].toMultiPickList).asPredictor

  val age = FeatureBuilder.Real[Passenger]
    .extract(d => Option(Double.unbox(d.getAge)).toReal).asPredictor

  val sibSp = FeatureBuilder.MultiPickList[Passenger]
    .extract(d => Option(d.getSibSp).map(_.toString).toSet[String].toMultiPickList).asPredictor

  val parch = FeatureBuilder.MultiPickList[Passenger]
    .extract(d => Option(d.getParch).map(_.toString).toSet[String].toMultiPickList).asPredictor

  val ticket = FeatureBuilder.MultiPickList[Passenger]
    .extract(d => Option(d.getTicket).toSet[String].toMultiPickList).asPredictor

  val fare = FeatureBuilder.Real[Passenger]
    .extract(d => Option(Double.unbox(d.getFare)).toReal).asPredictor

  val cabin = FeatureBuilder.MultiPickList[Passenger]
    .extract(d => Option(d.getCabin).toSet[String].toMultiPickList).asPredictor

  val embarked = FeatureBuilder.MultiPickList[Passenger]
    .extract(d => Option(d.getEmbarked).toSet[String].toMultiPickList).asPredictor

}
