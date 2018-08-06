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

package com.salesforce.op

import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.features.{FeatureUID, TransientFeature}

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
    val origins = inputs.flatMap(_.originFeatures).distinct.sorted.mkString("-").replaceAll("[^A-Za-z-]+", "")
    val stages = inputs.flatMap(_.stages).distinct.length + numberOperations
    s"${origins}_$stages-stagesApplied_$featureUid"
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

  /**
   * Replace feature names for multistage spark wrappers (gross-ness) that have had the output name overriden
   *
   * @param newName          override name
   * @param oldName          default name
   * @param numberOperations number of stages applied to add to stages from features (used in multistage operations)
   * @return name of the output ex: "f1-f2_4-stagesApplied_Real_0183nsdk"
   */
  private[stages] def updateOutputName(
    newName: String, oldName: String, numberOperations: Int
  ): String = {
    val name = raw"([A-Za-z-]+)_([0-9]+)-stagesApplied_([A-Za-z0-9_-]+)".r
    (newName, oldName) match {
      // old uid because stages are different
      case (nn, on) if nn == on => nn
      case (name(orig, sz, _), name(_, _, uid)) => s"${orig}_${sz.toInt + numberOperations}-stagesApplied_$uid"
      case (_, _) => throw new IllegalArgumentException(s"Cannot override output name $oldName with $newName")
    }
  }
}
