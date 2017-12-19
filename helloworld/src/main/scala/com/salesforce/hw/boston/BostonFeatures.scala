/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.hw.boston

import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._

trait BostonFeatures extends Serializable {

  val rowId = FeatureBuilder.Integral[BostonHouse].extract(_.rowId.toIntegral).asPredictor
  val crim = FeatureBuilder.RealNN[BostonHouse].extract(_.crim.toRealNN).asPredictor
  val zn = FeatureBuilder.RealNN[BostonHouse].extract(_.zn.toRealNN).asPredictor
  val indus = FeatureBuilder.RealNN[BostonHouse].extract(_.indus.toRealNN).asPredictor
  val chas = FeatureBuilder.MultiPickList[BostonHouse]
    .extract(x => Option(x.chas).toSet[String].toMultiPickList).asPredictor
  val nox = FeatureBuilder.RealNN[BostonHouse].extract(_.nox.toRealNN).asPredictor
  val rm = FeatureBuilder.RealNN[BostonHouse].extract(_.rm.toRealNN).asPredictor
  val age = FeatureBuilder.RealNN[BostonHouse].extract(_.age.toRealNN).asPredictor
  val dis = FeatureBuilder.RealNN[BostonHouse].extract(_.dis.toRealNN).asPredictor
  val rad = FeatureBuilder.Integral[BostonHouse].extract(_.rad.toIntegral).asPredictor
  val tax = FeatureBuilder.RealNN[BostonHouse].extract(_.tax.toRealNN).asPredictor
  val ptratio = FeatureBuilder.RealNN[BostonHouse].extract(_.ptratio.toRealNN).asPredictor
  val b = FeatureBuilder.RealNN[BostonHouse].extract(_.b.toRealNN).asPredictor
  val lstat = FeatureBuilder.RealNN[BostonHouse].extract(_.lstat.toRealNN).asPredictor

  val medv = FeatureBuilder.RealNN[BostonHouse].extract(_.medv.toRealNN).asResponse

}
