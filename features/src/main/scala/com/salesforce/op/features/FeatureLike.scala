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

package com.salesforce.op.features

import com.salesforce.op.FeatureHistory
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages._
import org.apache.spark.sql.types.Metadata
import org.slf4j.LoggerFactory
import scalax.collection.GraphPredef._
import scalax.collection._

import scala.reflect.runtime.universe.WeakTypeTag
import scala.util.{Failure, Success, Try}

/**
 * Feature definition
 *
 * @tparam O feature value type
 */
trait FeatureLike[O <: FeatureType] {

  /**
   * Feature name
   */
  val name: String

  /**
   * Unique identifier of the feature instance
   */
  val uid: String

  /**
   * Is this feature a response or a predictor
   */
  val isResponse: Boolean

  /**
   * Origin stage which resulted into creating this feature (transformer or estimator)
   */
  val originStage: OpPipelineStage[O]

  /**
   * The input features of the origin stage
   */
  val parents: Seq[OPFeature]

  /**
   * Weak type tag of the feature type O
   */
  implicit val wtt: WeakTypeTag[O]

  /**
   * A handy logger instance
   */
  @transient protected lazy val log = LoggerFactory.getLogger(this.getClass)

  /**
   * The distribution information of the feature
   * (is a sequence because map features have distribution for each key)
   */
  val distributions: Seq[FeatureDistributionLike]

  /**
   * The metadata to include in the raw feature generated
   */
  val metadata: Option[Metadata]

  /**
   * The distribution information of the feature computed during training
   * (is a sequence because map features have distribution for each key)
   */
  final def trainingDistributions: Seq[FeatureDistributionLike] =
    distributions.filter(_.`type` == FeatureDistributionType.Training)

  /**
   * The distribution information of the feature computed during scoring
   * (is a sequence because map features have distribution for each key)
   */
  final def scoringDistributions: Seq[FeatureDistributionLike] =
    distributions.filter(_.`type` == FeatureDistributionType.Scoring)

  /**
   * Check whether this feature's type [[O]] is a subtype of the given feature type [[T]]
   *
   * @tparam T the feature type to check
   * @return true if [[O]] conforms to [[T]], false otherwise
   */
  final def isSubtypeOf[T <: FeatureType : WeakTypeTag]: Boolean = FeatureType.isSubtype[O, T]

  /**
   * Feature type name
   *
   * @return feature type name
   */
  final def typeName: String = FeatureType.typeName[O](wtt)

  /**
   * Is this feature is raw or not (i.e. has no parent features)
   *
   * @return true if the feature is raw, false otherwise
   */
  final def isRaw: Boolean = parents.isEmpty

  /**
   * Convert this feature to json string
   *
   * @param pretty should pretty print
   * @return json string for feature
   */
  final def toJson(pretty: Boolean = true): String = FeatureJsonHelper.toJsonString(this, pretty = pretty)

  /**
   * Tests the equality of the FeatureLike objects
   *
   * Origin Stage is tested by uid
   *
   * Parents are test by uid and order dependent. This is because they are used as inputs to the origin stage
   * and input parameters may not be commutative
   */
  final override def equals(in: Any): Boolean = in match {
    case f: FeatureLike[O] => name == f.name && sameOrigin(f) && parents.map(_.uid) == f.parents.map(_.uid)
    case _ => false
  }

  /**
   * Tests the equality of the FeatureLike objects
   *
   * Origin Stage is tested by uid
   *
   * Parents are test by uid and order dependent. This is because they are used as inputs to the origin stage
   * and input parameters may not be commutative
   */
  final def sameOrigin(in: Any): Boolean = in match {
    case f: FeatureLike[O] =>
      isResponse == f.isResponse &&
        wtt.tpe =:= f.wtt.tpe && {
        originStage -> f.originStage match {
          case (null, null) => true
          case (null, _) => false
          case (_, null) => false
          case (os, fos) => os.uid == fos.uid
        }
      }
    case _ => false
  }

  /**
   * Returns the hash code of this feature
   *
   * @return hash code
   */
  final override def hashCode: Int = uid.hashCode

  final override def toString: String = {
    val valStr = Seq(
      "name" -> name,
      "uid" -> uid,
      "isResponse" -> isResponse,
      "originStage" -> Option(originStage).map(_.uid).orNull,
      "parents" -> parents.map(_.uid).mkString("[", ",", "]"),
      "distributions" -> distributions.map(_.toString).mkString("[", ",", "]"),
      "metadata" -> metadata
    ).map { case (n, v) => s"$n = $v" }.mkString(", ")

    s"${getClass.getSimpleName}($valStr)"
  }

  /**
   * Construct a raw feature instance from this feature which can be applied on a Dataframe.
   * Use this functionality when stacking workflows, e.g. when some features of a workflow
   * are used as raw or input features of another workflow.
   *
   * @param isResponse should make response or a predictor feature
   * @return new raw feature of the same type
   */
  final def asRaw(isResponse: Boolean = isResponse): FeatureLike[O] = {
    val raw = FeatureBuilder.fromRow[O](name)(wtt)
    if (isResponse) raw.asResponse else raw.asPredictor
  }

  /**
   * Transform the feature with a given transformation stage and an input feature
   *
   * @param stage transformer/estimator
   * @tparam U output type
   * @return transformed feature
   */
  final def transformWith[U <: FeatureType](
    stage: OpPipelineStage1[O, U]
  ): FeatureLike[U] = {
    stage.setInput(this).getOutput()
  }

  /**
   * Transform the feature with a given transformation stage and input features
   *
   * @param stage transformer/estimator
   * @param f     other feature
   * @tparam I other feature input type
   * @tparam U output type
   * @return transformed feature
   */
  final def transformWith[I <: FeatureType, U <: FeatureType](
    stage: OpPipelineStage2[O, I, U], f: FeatureLike[I]
  ): FeatureLike[U] = {
    stage.setInput(this, f).getOutput()
  }

  /**
   * Transform the feature with a given transformation stage and input features
   *
   * @param stage transformer/estimator
   * @param f1    other feature1
   * @param f2    other feature2
   * @tparam I1 f1 input type
   * @tparam I2 f2 input type
   * @tparam U  output type
   * @return transformed feature
   */
  final def transformWith[I1 <: FeatureType, I2 <: FeatureType, U <: FeatureType](
    stage: OpPipelineStage3[O, I1, I2, U], f1: FeatureLike[I1], f2: FeatureLike[I2]
  ): FeatureLike[U] = {
    stage.setInput(this, f1, f2).getOutput()
  }

  /**
   * Transform the feature with a given transformation stage and input features
   *
   * @param stage transformer/estimator
   * @param f1    other feature1
   * @param f2    other feature2
   * @param f3    other feature2
   * @tparam I1 f1 input type
   * @tparam I2 f2 input type
   * @tparam I3 f3 input type
   * @tparam U  output type
   * @return transformed feature
   */
  final def transformWith[I1 <: FeatureType, I2 <: FeatureType, I3 <: FeatureType, U <: FeatureType](
    stage: OpPipelineStage4[O, I1, I2, I3, U], f1: FeatureLike[I1], f2: FeatureLike[I2], f3: FeatureLike[I3]
  ): FeatureLike[U] = {
    stage.setInput(this, f1, f2, f3).getOutput()
  }

  /**
   * Transform the feature with a given transformation stage and input features
   *
   * @param stage transformer/estimator
   * @param fs    other features
   * @tparam U output type
   * @return transformed feature
   */
  final def transformWith[U <: FeatureType](
    stage: OpPipelineStageN[O, U], fs: Array[FeatureLike[O]]
  ): FeatureLike[U] = {
    stage.setInput(this +: fs).getOutput()
  }

  /**
   * History of all stages and origin features used to create a given feature
   *
   * @return [[FeatureHistory]] containing all feature history information
   */
  final def history(): FeatureHistory = {
    // TODO see if we need to make this more efficient
    val originFeatures = rawFeatures.map(_.name).distinct.sorted
    val stages = parentStages(verbose = false) match {
      case Failure(err) => throw new IllegalArgumentException(s"Failed to compute parent stages for feature $uid", err)
      case Success(res) => res
    }
    val stageNames = stages.toSeq
      .map { case (stage, distance) => distance -> stage.stageName }
      .sortBy { case (distance, stageName) => -distance -> stageName }
      .map { case (_, stageName) => stageName }
    FeatureHistory(originFeatures = originFeatures, stages = stageNames)
  }


  /**
   * Traverses over the feature dependency graph (using DFS)
   *
   * @param acc accumulator
   * @param f   accumulate function
   * @tparam T accumulator item type
   * @return accumulator
   */
  private[op] final def traverse[T](acc: T)(f: (T, OPFeature) => T): T = {
    def dfs(
      t: OPFeature,
      visited: Set[OPFeature]
    ): Set[OPFeature] = {
      if (visited.contains(t)) visited
      else {
        val next = t.parents.filterNot(visited.contains)
        next.foldLeft(visited + t)((acc, n) => dfs(n, acc))
      }
    }

    dfs(this, Set.empty).foldLeft(acc)(f)
  }

  /**
   * Collects all the features in feature dependency graph
   *
   * @return all features
   */
  private[op] final def allFeatures: List[OPFeature] = {
    traverse(List.empty[OPFeature])((acc, f) => f :: acc).distinct
  }

  /**
   * Collects all the raw features in feature dependency graph
   *
   * @return all the raw features
   */
  private[op] final def rawFeatures: List[OPFeature] = {
    traverse(List.empty[OPFeature])((acc, f) => if (f.isRaw) f :: acc else acc).distinct
  }

  /**
   * Checks that the parent features for all features passed in match the input features set on the origin stage of the
   * feature. This is to guard against attempts to reuse stages
   *
   * @param features all features being used
   * @return boolean if all features have parents that match their origin stage inputs
   */
  private[op] final def checkFeatureOriginStageMatch(features: Iterable[OPFeature]): Boolean = {
    features.filterNot(_.isRaw).forall { f =>
      val parentStageInputs = f.originStage.getInputFeatures().map(_.uid).toSet
      val featureInputs = f.parents.map(_.uid).toSet
      parentStageInputs.diff(featureInputs).isEmpty
    }
  }

  /**
   * Computes the longest distances from this feature to all the parent stages (using Topological Sort)
   *
   * @param verbose should log the parent stages graph construction
   * @return feature's parent stages with longest distances
   */
  private[op] final def parentStages(verbose: Boolean = true): Try[Map[OPStage, Int]] = Try {
    val featureParents =
      traverse(List.empty[(OPFeature, OPFeature)])((acc, f) =>
        f.parents.map(p => f -> p).toList ::: acc
      )

    val allFeatures = featureParents.map(_._1) ++ featureParents.map(_._2)
    val featuresByUid =
      allFeatures.foldLeft(Map.empty[String, OPFeature])((acc, f) =>
        if (acc.contains(f.uid)) acc else acc + (f.uid -> f)
      )

    require(checkFeatureOriginStageMatch(featuresByUid.values), "Some of your features had parent features that did" +
      " not match the inputs to their origin stage. All stages must be a new instance when used to transform features")

    def logDebug(msg: String) = log.debug(s"[${this.uid}]: $msg")

    if (log.isDebugEnabled && verbose) {
      logDebug("*" * 80)
      logDebug(s"Collected ${featuresByUid.keys.size} unique features with ${featureParents.size} parent stages.")
      featuresByUid.foreach(f => logDebug(s"Feature [${f._1}] - ${f._2.name}"))
      logDebug("*" * 80)
      featureParents.foreach(f => logDebug(s"Feature -> Parent: ${f._1.uid} -> ${f._2.uid}"))
      logDebug("*" * 80)
    }

    val graph =
      featureParents.map { case (f1, f2) => f1.uid ~> f2.uid }
        .foldLeft(Graph.empty[String, GraphEdge.DiEdge])(_ + _)

    if (log.isDebugEnabled && verbose) {
      logDebug("*" * 80)
      logDebug(s"Constructed a graph with ${graph.nodes.size} nodes and ${graph.edges.size} edges.")
      logDebug("Nodes: " + graph.nodes.map(_.value).toArray.sorted.mkString(", "))
      graph.edges.foreach(e => logDebug("Edge: " + e.map(_.value).mkString(" -> ")))
      logDebug("*" * 80)
      logDebug("*" * 80)
      logDebug(s"The layered graph sorted with Topological Sort.")
    }

    val tSorted: Map[OPStage, Int] =
      graph.topologicalSort match {
        case Left(node) => throw new FeatureCycleException(from = this, to = featuresByUid(node.value))
        case Right(sorted) =>
          (for {
            (distance, nodes) <- sorted.toLayered
            node <- nodes
            feature = featuresByUid(node.value)
            stage = feature.originStage
            if stage != null && !stage.isInstanceOf[FeatureGeneratorStage[_, _]]
          } yield {
            if (log.isDebugEnabled) {
              logDebug(s"${feature.name} with origin stage $stage with distance $distance")
            }
            stage -> distance
          }).toMap
      }

    if (log.isDebugEnabled && verbose) {
      logDebug("*" * 80)
    }
    tSorted
  }

  /**
   * Pretty print feature's parent stages tree
   *
   * @return feature's parent stages tree with indentation
   */
  final def prettyParentStages: String = {
    val sb = new StringBuilder
    val stack = new scala.collection.mutable.Stack[(Int, OPFeature)]
    stack.push((0, this))
    while (stack.nonEmpty) {
      val (indentLevel, elem) = stack.pop()
      if (elem.originStage != null) {
        sb.append(s"${"|    " * indentLevel}+-- ${elem.originStage.operationName}\n")
        elem.parents.foreach(e => stack.push((indentLevel + 1, e)))
      }
    }
    sb.mkString
  }

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
  private[op] def copyWithNewStages(stages: Array[OPStage]): FeatureLike[O]

  /**
   * Takes an a sequence of feature distributions associated with the feature
   *
   * @param distributions Seq of the feature distributions for the feature
   * @return A feature with the distributions associated
   */
  private[op] def withDistributions(distributions: Seq[FeatureDistributionLike]): FeatureLike[O]


  /**
   * Adds metadata to feature so can be included in extracted dataframe
   *
   * @param metadataIn dataframe metadata to include in the
   * @return A feature with the metadata associated
   */
  def withMetadata(metadataIn: Metadata): FeatureLike[O]
}
