/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
