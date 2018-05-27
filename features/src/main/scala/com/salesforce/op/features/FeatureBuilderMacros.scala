/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.features

import com.salesforce.op.features.types._

import scala.language.experimental.macros
import scala.reflect.macros.blackbox

/**
 * Feature Builder macros to capture feature extract method source code
 */
private[features] object FeatureBuilderMacros {

  // scalastyle:off

  def extract[I: c.WeakTypeTag, O <: FeatureType : c.WeakTypeTag]
  (c: blackbox.Context)(fn: c.Expr[I => O]): c.Expr[FeatureBuilderWithExtract[I, O]] = {
    import c.universe._
    val name = reify(c.prefix.splice.asInstanceOf[FeatureBuilder[_, _]].name)
    val fnSource = ParamRename(c).transform(fn.tree).toString()
    c.Expr(
      q"""
          new com.salesforce.op.features.FeatureBuilderWithExtract($name, $fn, $fnSource)
      """
    )
  }

  def extractWithDefault[I: c.WeakTypeTag, O <: FeatureType : c.WeakTypeTag]
  (c: blackbox.Context)(fn: c.Expr[I => O], default: c.Expr[O]): c.Expr[FeatureBuilderWithExtract[I, O]] = {
    import c.universe._
    val in = symbolOf[I]
    val fnWithDefault = c.Expr[I => O](
      q"""
         (in: $in) => try $fn(in) catch { case _: Exception => $default }
      """
    )
    extract[I, O](c)(fnWithDefault)
  }

  /**
   * Traverses code tree and renames all parameters with name of the form: 'x$i', where 'i' a counter starting with 0.
   */
  private object ParamRename {
    def apply(c: blackbox.Context) = {
      import c.universe._
      new Transformer {
        var i = 0
        var renames = Map[String, String]()
        override def transform(tree: Tree): Tree = {
          val nt = tree match {
            case ValDef(m, TermName(v), tpe, tr) if m.hasFlag(Flag.PARAM) =>
              val newName = "x$" + i.toString
              i = i + 1
              renames = renames + (v -> newName)
              ValDef(m, TermName(newName), tpe, tr)
            case Ident(TermName(v)) if renames.contains(v) => Ident(TermName(renames(v)))
            case x => x
          }
          super.transform(nt)
        }
      }
    }
  }

  // scalastyle:on
}
