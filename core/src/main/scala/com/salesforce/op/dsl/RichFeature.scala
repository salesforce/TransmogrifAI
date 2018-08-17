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

package com.salesforce.op.dsl

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.BinaryLambdaTransformer
import com.salesforce.op.stages.base.quaternary.QuaternaryLambdaTransformer
import com.salesforce.op.stages.base.ternary.TernaryLambdaTransformer
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.stages.impl.feature.{AliasTransformer, ToOccurTransformer}
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams

import scala.reflect.runtime.universe.TypeTag

trait RichFeature {

  /**
   * Enrichment functions for Feature[A]
   *
   * @param feature FeatureLike
   * @tparam A input type
   */
  implicit class RichFeature[A <: FeatureType : TypeTag]
  (val feature: FeatureLike[A])(implicit val ftt: TypeTag[A#Value]) {

    /**
     * Unary transform feature[A] => feature[B]
     *
     * @param f map A => B
     * @return feature of type B
     */
    def map[B <: FeatureType : TypeTag](f: A => B, operationName: String = "map")
      (implicit ttb: TypeTag[B#Value]): FeatureLike[B] = {
      feature.transformWith(
        new UnaryLambdaTransformer[A, B](operationName = operationName, transformFn = f)
      )
    }

    /**
     * Replace a matching value with a new one
     *
     * @param oldVal of type A
     * @param newVal of type A
     * @return feature of type A
     */
    def replaceWith(oldVal: A, newVal: A): FeatureLike[A] = {
      map[A](a => if (oldVal == a) newVal else a)
    }

    /**
     * Binary transform (feature[A], feature[B]) => feature[C]
     *
     * @param f map (A, B) => C
     * @return feature of type C
     */
    def map[B <: FeatureType : TypeTag, C <: FeatureType : TypeTag](
      f1: FeatureLike[B], f: (A, B) => C
    )(implicit ttb: TypeTag[B#Value], ttc: TypeTag[C#Value]): FeatureLike[C] = {
      feature.transformWith(
        new BinaryLambdaTransformer[A, B, C](operationName = "map", transformFn = f),
        f = f1
      )
    }

    /**
     * Ternary transform (feature[A], feature[B], feature[C]) => feature[D]
     *
     * @param f map (A, B, C) => D
     * @return feature of type D
     */
    def map[B <: FeatureType : TypeTag, C <: FeatureType : TypeTag, D <: FeatureType : TypeTag](
      f1: FeatureLike[B], f2: FeatureLike[C], f: (A, B, C) => D
    )(implicit ttb: TypeTag[B#Value], ttc: TypeTag[C#Value], ttd: TypeTag[D#Value]): FeatureLike[D] = {
      feature.transformWith(
        new TernaryLambdaTransformer[A, B, C, D](operationName = "map", transformFn = f),
        f1 = f1, f2 = f2
      )
    }

    /**
     * Quaternary transform (feature[A], feature[B], feature[C], feature[D]) => feature[E]
     *
     * @param f map (A, B, C, D) => E
     * @return feature of type E
     */
    def map[B <: FeatureType : TypeTag,
    C <: FeatureType : TypeTag, D <: FeatureType : TypeTag, E <: FeatureType : TypeTag](
      f1: FeatureLike[B], f2: FeatureLike[C], f3: FeatureLike[D], f: (A, B, C, D) => E
    )(implicit ttb: TypeTag[B#Value], ttc: TypeTag[C#Value], ttd: TypeTag[D#Value], tte: TypeTag[E#Value]
    ): FeatureLike[E] = {
      feature.transformWith(
        new QuaternaryLambdaTransformer[A, B, C, D, E](operationName = "map", transformFn = f),
        f1 = f1, f2 = f2, f3 = f3
      )
    }

    /**
     * Filter feature[A] using the predicate.
     * Filtered out values are replaced with a default.
     *
     * @param p       predicate A => Boolean
     * @param default default value if predicate returns false
     * @return feature of type A
     */
    def filter(p: A => Boolean, default: A): FeatureLike[A] = {
      feature.transformWith(
        new UnaryLambdaTransformer[A, A](operationName = "filter", transformFn = a => if (p(a)) a else default)
      )
    }

    /**
     * Filter feature[A] using the NOT predicate.
     * Filtered out values are replaced with a default.
     *
     * @param p       predicate A => Boolean
     * @param default default value if predicate returns false
     * @return feature of type A
     */
    def filterNot(p: A => Boolean, default: A): FeatureLike[A] = {
      filter(a => !p(a), default)
    }

    /**
     * Filter & transform feature[A] => feature[B] using the partial function A => B.
     * Filtered out values are replaced with a default.
     *
     * @param default default value if partial function is not defined
     * @param pf      partial function A => B
     * @return feature of type B
     */
    def collect[B <: FeatureType : TypeTag](default: B)(pf: PartialFunction[A, B])
      (implicit ttb: TypeTag[B#Value]): FeatureLike[B] = {
      feature.transformWith(
        new UnaryLambdaTransformer[A, B](
          operationName = "collect",
          transformFn = a => if (pf.isDefinedAt(a)) pf(a) else default
        )
      )
    }

    /**
     * Tests whether a predicate holds for feature[A]
     *
     * @param p predicate to apply on feature[A]
     * @return feature[Binary]
     */
    def exists(p: A => Boolean): FeatureLike[Binary] = {
      feature.transformWith(
        new UnaryLambdaTransformer[A, Binary](
          operationName = "exists",
          transformFn = a => new Binary(p(a))
        )
      )
    }

    /**
     * Apply ToOccur transformer shortcut function
     *
     * @return transformed feature of type Numeric (doolean)
     */
    def occurs(): FeatureLike[RealNN] = {
      feature.transformWith(new ToOccurTransformer[A]())
    }

    /**
     * Apply ToOccur transformer shortcut function
     * This version allows a user to specify a non default matchFn
     *
     * @return transformed feature of type Numeric (doolean)
     */
    def occurs(matchFn: A => Boolean): FeatureLike[RealNN] = {
      feature.transformWith(new ToOccurTransformer[A](matchFn = matchFn))
    }

    /**
     * Create an alias of this feature by capturing the val name (note will not work on raw features)
     * @return alias of the feature
     */
    def alias(implicit name: sourcecode.Name): FeatureLike[A] = alias(name = name.value)

    /**
     * Create an alias of this feature with the desired name (note will not work on raw features)
     * @param name desired feature name
     * @return alias of the feature
     */
    def alias(name: String): FeatureLike[A] = {
      feature.originStage match {
        case _: SparkWrapperParams[_] => feature.transformWith(new AliasTransformer(name))
        case _ if feature.isRaw => feature.transformWith(new AliasTransformer(name))
        case s => s.setOutputFeatureName(name).getOutput()
      }
    }

  }

}
