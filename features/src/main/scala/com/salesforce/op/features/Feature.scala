/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features

import com.salesforce.op.UID
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.{OPStage, OpPipelineStage}

import scala.reflect.runtime.universe.WeakTypeTag

/**
 * Feature instance
 *
 * Note: can only be created using [[FeatureBuilder]].
 *
 * @param name        name of feature, represents the name of the column in the dataframe.
 * @param isResponse  whether or not this feature is a response feature, ie dependent variable
 * @param originStage reference to OpPipelineStage responsible for generating the feature.
 * @param parents     references to the features that are transformed by the originStage that produces this feature
 * @param uid         unique identifier of the feature instance
 * @param wtt         feature's value type tag
 * @tparam O feature value type
 */
private[op] case class Feature[O <: FeatureType]
(
  name: String,
  isResponse: Boolean,
  originStage: OpPipelineStage[O],
  parents: Seq[OPFeature],
  uid: String
)(implicit val wtt: WeakTypeTag[O]) extends FeatureLike[O] {

  def this(
    name: String,
    isResponse: Boolean,
    originStage: OpPipelineStage[O],
    parents: Seq[OPFeature]
  )(implicit wtt: WeakTypeTag[O]) = this(
    name = name,
    isResponse = isResponse,
    originStage = originStage,
    parents = parents,
    uid = FeatureUID(originStage.uid)
  )(wtt)

  /**
   * Takes an array of stages and will try to replace all origin stages of features with
   * stage from the new stages with the same uid. This is used to make a copy of the feature
   * with the origin stage pointing at the fitted model resulting from an estimator rather
   * than the estimator.
   *
   * @param stages Array of all parent stages for the features
   * @return A feature with the origin stage (and the origin stages or all parent stages replaced
   *         with the stages in the map passed in)
   */
  private[op] final def copyWithNewStages(stages: Array[OPStage]): FeatureLike[O] = {
    val stagesMap: Map[String, OpPipelineStage[_]] = stages.map(s => s.uid -> s).toMap

    def copy[T <: FeatureType](f: FeatureLike[T]): Feature[T] = {
      // try to get replacement stage, if no replacement provided use original
      val stage = stagesMap.getOrElse(f.originStage.uid, f.originStage).asInstanceOf[OpPipelineStage[T]]
      val newParents = f.parents.map(p => copy[T](p.asInstanceOf[FeatureLike[T]]))
      Feature[T](
        name = f.name, isResponse = f.isResponse, originStage = stage, parents = newParents, uid = f.uid
      )(f.wtt)
    }

    copy(this)
  }

}

/**
 * Feature UID factory
 */
case object FeatureUID {

  /**
   * Returns a UID for features that is built of: feature type name + "_" + 12 hex chars of stage uid.
   *
   * @tparam T feature type T with a type tag
   * @param stageUid stage uid
   * @return UID
   */
  def apply[T <: FeatureType : WeakTypeTag](stageUid: String): String = {
    val (_, stageUidSuffix) = UID.fromString(stageUid)
    val shortTypeName = FeatureType.shortTypeName[T]
    s"${shortTypeName}_$stageUidSuffix"
  }

}
