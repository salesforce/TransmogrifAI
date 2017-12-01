/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op

import com.salesforce.op.features.{FeatureUID, TransientFeature}
import com.salesforce.op.features.types.FeatureType

import scala.reflect.runtime.universe._


package object stages {

  /**
   * Some common type shortcuts
   */
  type OPStage = OpPipelineStage[_ <: FeatureType]

  /**
   * Generate the name of the output feature for multioutput stages
   *
   * @param featureUid       feature uid (for uniqueness)
   * @param inputs           transient features for inputs
   * @param numberOperations number of stages applied to add to stages from features (used in multistage operations)
   * @return name of the output ex: "f1-f2_4-stagesApplied_Real_0183nsdk"
   */
  private[stages] def makeOutputName(
    featureUid: String, inputs: Seq[TransientFeature], numberOperations: Int = 1
  ): String = {
    val origins = inputs.flatMap(_.originFeatures).distinct.sorted
    val stages = inputs.flatMap(_.stages).distinct.length + numberOperations
    s"${origins.mkString("-")}_$stages-stagesApplied_$featureUid"
  }

  /**
   * Generate the name of the output feature for multi output stages
   *
   * @param stageUid    stage uid used to get feature UID
   * @param inputs      transient features for inputs
   * @param numberOperations number of stages applied to add to stages from features (used in multistage operations)
   * @return name of the output ex: "f1-f2_4-stagesApplied_Real_0183nsdk"
   */
  private[stages] def makeOutputNameFromStageId[T <: FeatureType : WeakTypeTag](
    stageUid: String, inputs: Seq[TransientFeature], numberOperations: Int = 1
  ): String = {
    val featureUid = FeatureUID[T](stageUid)
    makeOutputName(featureUid, inputs, numberOperations)
  }
}
