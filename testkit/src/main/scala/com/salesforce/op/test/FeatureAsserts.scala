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

import com.salesforce.op.aggregators.{Event, MonoidAggregatorDefaults}
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.FeatureGeneratorStage
import com.twitter.algebird.MonoidAggregator
import org.joda.time.Duration
import org.scalatest.Matchers

import scala.reflect.runtime.universe.WeakTypeTag

/**
 * Asserts for Feature instances on a given input/output
 */
trait FeatureAsserts extends Matchers {

  /**
   * Assert Feature instance on a given input/output
   *
   * @param f               feature to assert
   * @param in              input value
   * @param out             expected output value
   * @param name            expected name
   * @param isResponse      is expected to be a response
   * @param aggregator      expected aggregator
   * @param aggregateWindow expected aggregate window
   * @param tti             expected input typetag
   * @param wtt             expected output typetag
   * @tparam I input type
   * @tparam O output feature type
   */
  def assertFeature[I, O <: FeatureType](f: FeatureLike[O])(
    in: I, out: O, name: String, isResponse: Boolean = false,
    aggregator: WeakTypeTag[O] => MonoidAggregator[Event[O], _, O] =
    (wtt: WeakTypeTag[O]) => MonoidAggregatorDefaults.aggregatorOf[O](wtt),
    aggregateWindow: Option[Duration] = None
  )(implicit tti: WeakTypeTag[I], wtt: WeakTypeTag[O]): Unit = {
    f.name shouldBe name
    f.isResponse shouldBe isResponse
    f.parents shouldBe Nil
    f.uid.startsWith(wtt.tpe.dealias.toString.split("\\.").last) shouldBe true
    f.wtt.tpe =:= wtt.tpe shouldBe true
    f.isRaw shouldBe true
    f.typeName shouldBe wtt.tpe.typeSymbol.fullName

    f.originStage shouldBe a[FeatureGeneratorStage[_, _ <: FeatureType]]
    val fg = f.originStage.asInstanceOf[FeatureGeneratorStage[I, O]]
    fg.tti shouldBe tti
    fg.aggregator shouldBe aggregator(wtt)
    fg.extractFn(in) shouldBe out
    fg.extractSource.nonEmpty shouldBe true // TODO we should eval the code here: eval(fg.extractSource)(in)
    fg.getOutputFeatureName shouldBe name
    fg.outputIsResponse shouldBe isResponse
    fg.aggregateWindow shouldBe aggregateWindow
    fg.uid.startsWith(classOf[FeatureGeneratorStage[I, O]].getSimpleName) shouldBe true
  }

}
