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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer

import scala.reflect.runtime.universe.TypeTag

/**
 * Transformer that converts input feature of type I into doolean feature using a user specified function that
 * maps object type I to a Boolean
 *
 * @param uid     uid for instance
 * @param matchFn A function that allows the user to pass in a function that maps object I to a Boolean.
 * @tparam I Object type to be mapped to a double (doolean).
 */
class ToOccurTransformer[I <: FeatureType]
(
  uid: String = UID[ToOccurTransformer[I]],
  val matchFn: I => Boolean = new ToOccurTransformer.DefaultMatches[I]
)(implicit tti: TypeTag[I])
  extends UnaryTransformer[I, RealNN](operationName = "toOccur", uid = uid) {

  private val (yes, no) = (RealNN(1.0), RealNN(0.0))

  def transformFn: I => RealNN = (value: I) => if (matchFn(value)) yes else no

}


object ToOccurTransformer {
  class DefaultMatches[T <: FeatureType] extends Function1[T, Boolean] with Serializable {
    def apply(t: T): Boolean = t match {
      case num: OPNumeric[_] if num.nonEmpty => num.toDouble.get > 0.0
      case text: Text if text.nonEmpty => text.value.get.length > 0
      case collection: OPCollection => collection.nonEmpty
      case _ => false
    }
  }

  def defaultMatches[T <: FeatureType]: T => Boolean = new DefaultMatches[T]
}
