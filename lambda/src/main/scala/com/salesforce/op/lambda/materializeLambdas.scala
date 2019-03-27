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

package com.salesforce.op.lambda

import scala.reflect.macros.blackbox
import scala.language.experimental.macros
import scala.annotation.{StaticAnnotation, compileTimeOnly}
import scala.collection.mutable.ListBuffer

@compileTimeOnly("enable macro paradise to expand macro annotations")
class materializeLambdas extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro MaterializeLambdasImpl.impl
}

object MaterializeLambdasImpl {
  def impl(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._

    val inputs = annottees.map(_.tree).toList

    class ValDefTransformer(name: String) extends Transformer {
      var i = 0
      val functions = ListBuffer[Tree]()
      val valPrefix: String = s"auto_val_${name}"

      def getValDefName() = {
        val fncName = s"fnc${i}"
        i += 1
        TermName(s"${valPrefix}_$fncName")
      }

      override def transform(tree: Tree): Tree = {
        tree match {
          case q"((..$params => $body): $t1 => $t2)" => {

            val n = getValDefName
            val transformedBody = transform(body.asInstanceOf[c.Tree])
            // scalastyle:off
            functions.append(q"def $n = new Function1[${t1},${t2}]{ override def apply(${params.head}) = ${transformedBody} }")
            // scalastyle:on
            q"Lambdas.${n}"

          }

          case q"(..$params => $body)" => {

            val n = getValDefName
            val transformedBody = transform(body.asInstanceOf[c.Tree])
            functions.append(q"def $n = (..${params}) => ${transformedBody} ")
            q"Lambdas.${n}"

          }

          case x => super.transform(x)
        }
      }
    }

    class ClassTransformer extends Transformer {
      val originalLambdas = ListBuffer[Tree]()

      def valMacroMatch(x: c.Tree) = x match {
        case q"new lambdas()" => true
        case _ => false
      }

      def objMacroMatch(x: c.Tree) = x match {
        case q"new materializeLambdas()" => true
        case _ => false
      }

      override def transform(tree: Tree): Tree = {
        tree match {
          // this is needed in order to prevent analyzing vals from nested objects
          case ClassDef(mods, _, _, _) if mods.annotations.exists(objMacroMatch) => {
            tree
          }
          case ModuleDef(mods, _, _) if mods.annotations.exists(objMacroMatch) => {
            tree
          }
          case ValDef(mods, vName, _, expr) if mods.annotations.exists(valMacroMatch) => {
            val t = new ValDefTransformer(vName.toString)
            val res = t.transform(expr.asInstanceOf[Tree])
            originalLambdas ++= t.functions
            q"val ${vName} = ${res}"
          }

          case x => super.transform(x)
        }
      }
    }

    val t = new ClassTransformer()

    val res = t.transform(inputs.head)
    val objDef = q"""object Lambdas extends Serializable { ..${t.originalLambdas} }"""
    val objs = inputs match {

      // class/trait present with companion object
      case ClassDef(cmods, cname, ctparams, ctpl) :: ModuleDef(mods, name, Template(parents, self, body)) :: Nil => {
        res :: ModuleDef(mods, name, Template(parents, self, objDef :: body)) :: Nil
      }

      // only object present
      case ModuleDef(mods, name, Template(parents, self, body)) :: Nil => {
        val m = res.asInstanceOf[ModuleDef]
        ModuleDef(mods, name, Template(parents, self, objDef :: m.impl.body)) :: Nil
      }
      // scalastyle:off
      case _ => throw new Exception(s"Unsupported application of the macro (must be object or class with companion object)")
      // scalastyle:on
    }

    val withLambdas: List[c.Tree] = objs
    c.Expr[Any](Block(withLambdas, Literal(Constant(()))))
  }
}
