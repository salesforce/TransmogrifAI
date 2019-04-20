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

package com.salesforce.op.utils.reflection

import scala.reflect._
import scala.reflect.runtime.universe._
import scala.reflect.runtime.{universe => runtimeUniverse}
import scala.util.{Failure, Success}

/**
 * Various Reflection helpers
 */
object ReflectionUtils {

  /**
   * Default class loader - Thread.currentThread().getContextClassLoader
   */
  def defaultClassLoader: ClassLoader = Thread.currentThread().getContextClassLoader

  /**
   * Get a runtime mirror
   *
   * @param classLoader class loader to use
   * @return runtime mirror
   */
  def runtimeMirror(classLoader: ClassLoader = defaultClassLoader): Mirror = {
    runtimeUniverse.runtimeMirror(classLoader)
  }

  /**
   * Create a new instance of type T given a ctor args getter function
   *
   * @param klazz       instance class
   * @param ctorArgs    ctor args getter function
   * @param classLoader class loader to use
   * @tparam T
   * @return new instance of T
   */
  def newInstance[T](
    klazz: Class[_],
    ctorArgs: (String, Symbol) => util.Try[Any],
    classLoader: ClassLoader = defaultClassLoader
  ): T = {
    val (runtimeMirror, classMirror) = mirrors(klazz, classLoader)
    val classType = runtimeMirror.classSymbol(klazz).toType
    // reflect best constructor
    val (ctor, args) = bestCtor(klazz, classType, classMirror, ctorArgs)
    // apply the constructor on the extracted args
    ctor.apply(args.map(_._2): _*).asInstanceOf[T]
  }

  /**
   * Copy any instance using reflection.
   *
   * Note: all the implicit type values must be explicitly mentioned in the ctor.
   *
   * So instead of implicitly passing the TypeTag:
   * case class MyClass[T : TypeTag](t: T)
   *
   * Explicitly pass TypeTag as a val:
   * case class MyClass[T](t: T)(implicit val tt: TypeTag[T])
   *
   * @param instance instance of type T
   * @tparam T type T
   * @param classLoader class loader to use
   * @return a copy of t
   * @throws RuntimeException in case a ctor param value cannot be extracted
   */
  def copy[T: ClassTag](instance: T, classLoader: ClassLoader = defaultClassLoader): T = {
    val (ctor, args) = bestCtorWithArgs(instance, classLoader)
    ctor.apply(args.map(_._2): _*).asInstanceOf[T]
  }

  /**
   * Use reflection to get the primary constructor definition of an instance.
   * Along with the constructor Mirror, this will also search the instance mirror for
   * the appropriate members and their values used as arguments to the
   * constructor.
   *
   * @param instance    instance to make copy from
   * @param classLoader class loader to use
   * @tparam T type of instance to copy
   * @return a tuple where the first element is the Constructor Mirror,
   *         and the second element is a List of tuples containing the
   *         member names and their values
   */
  def bestCtorWithArgs[T: ClassTag](
    instance: T,
    classLoader: ClassLoader = defaultClassLoader
  ): (MethodMirror, List[(String, Any)]) = {
    val klazz = instance.getClass
    val (runtimeMirror, classMirror) = mirrors(klazz, classLoader)
    val classType = runtimeMirror.classSymbol(klazz).toType
    val tMembers = classType.members
    val gettrs = tMembers.collect { case m: MethodSymbol if m.isGetter && m.isPublic => termNameStr(m.name) -> m }.toMap
    val vals = tMembers.collect { case v: TermSymbol if v.isVal => termNameStr(v.name) -> v }.toMap
    val instanceMirror = runtimeMirror.reflect(instance)

    def ctorArgs(paramName: String, param: Symbol): util.Try[Any] = util.Try {
      val getterOpt = gettrs.get(paramName).map(instanceMirror.reflectMethod)
      if (getterOpt.isDefined) getterOpt.get.apply()
      else instanceMirror.reflectField(vals(paramName)).get
    }
    bestCtor(klazz, classType, classMirror, ctorArgs)
  }


  /**
   * Find setter methods for the provided method name
   * @param instance    class to find method for
   * @param setterName  name of method to find
   * @param args        argument values
   * @param argsCount   optional number of arguments to match
   * @param classLoader class loader to use
   * @tparam T type of instance to copy
   * @return reflected method to set type
   */
  def reflectSetterMethod[T: ClassTag](
    instance: T,
    setterName: String,
    args: Seq[Any],
    argsCount: Option[Int] = None,
    classLoader: ClassLoader = defaultClassLoader
  ): Any = {
    reflectMethod(instance, s"set$setterName", argsCount, classLoader).apply(args: _*)
  }

  /**
   * Find setter methods for the provided method name
   * @param instance    class to find method for
   * @param methodName  name of method to find
   * @param argsCount   optional number of arguments to match
   * @param classLoader class loader to use
   * @tparam T type of instance to copy
   * @return reflected method to set type
   */
  def reflectMethod[T: ClassTag](
    instance: T,
    methodName: String,
    argsCount: Option[Int] = None,
    classLoader: ClassLoader = defaultClassLoader
  ): MethodMirror = {
    val klazz = instance.getClass
    val (runtimeMirror, classMirror) = mirrors(klazz, classLoader)
    val classType = runtimeMirror.classSymbol(klazz).toType
    val tMembers = classType.members
    val methodsWithParams = tMembers.collect { case m: MethodSymbol => m -> m.paramLists.flatten }
    val methods = methodsWithParams.collect {
      case (m: MethodSymbol, params) if m.isMethod &&
        termNameStr(m.name).compareToIgnoreCase(methodName) == 0 &&
        (argsCount.isEmpty || argsCount.contains(params.length)) => m -> params
    }.toList.sortBy(-_._2.length).map(_._1)

    methods match {
      case method :: _ =>
        val instanceMirror = runtimeMirror.reflect(instance)
        instanceMirror.reflectMethod(method)
      case Nil =>
        throw new RuntimeException(s"Method with name '$methodName' was not found on instance of type: $klazz")
    }
  }

  /**
   * Find the class by name
   *
   * @param name        class name
   * @param classLoader class loader to use
   * @throws ClassNotFoundException
   * @return class object
   */
  def classForName(name: String, classLoader: ClassLoader = defaultClassLoader): Class[_] = classLoader.loadClass(name)


  /**
   * Fully dealiased type name for [[Type]].
   * This method performs a recursive dealising vs a regular type.dealias, which does on one level only.
   *
   * E.g: given a type of "Map[String,Double]" the result is
   * "scala.collection.immutable.Map[java.lang.String,scala.Double]"
   *
   * @param t type to dealias
   * @return fully dealised type name
   */
  def dealisedTypeName(t: Type): String = {
    val dealised = t.dealias
    if (dealised.typeArgs.isEmpty) dealised.typeSymbol.fullName
    else {
      dealised.typeConstructor.dealias.typeSymbol.fullName +
        dealised.typeArgs.map(dealisedTypeName).mkString("[", ",", "]")
    }
  }

  /**
   * Create a TypeTag for Type
   *
   * @param rtm runtime mirror
   * @param tpe type
   * @tparam T type T
   * @return TypeTag[T]
   */
  def typeTagForType[T](rtm: Mirror = runtimeMirror(), tpe: Type): TypeTag[T] = {
    TypeTag(rtm, new api.TypeCreator {
      def apply[U <: api.Universe with Singleton](m: api.Mirror[U]): U#Type =
        if (m eq rtm) tpe.asInstanceOf[U#Type]
        else throw new IllegalArgumentException(s"Type tag defined in $rtm cannot be migrated to other mirrors.")
    })
  }


  /**
   * Returns a Type Tag by string name
   *
   * @param rtm runtime mirror
   * @param n   class name
   * @return TypeTag[_]
   */
  def typeTagForName(rtm: Mirror = runtimeMirror(), n: String): TypeTag[_] = {
    val clazz = classForName(n)
    typeTagForType(rtm, rtm.classSymbol(clazz).toType)
  }

  /**
   * A helper function to get instance of lambda function or object
   * @param name full name
   * @return
   */
  def getInstanceOfObject[T](name: String): T = {
    val clazz = ReflectionUtils.classForName(name)

    val res = clazz.getConstructors.headOption match {
      case Some(c) => c.newInstance()
      case _ => {
        clazz.getField("MODULE$").get(clazz)
      }
    }
    res.asInstanceOf[T]
  }

  /**
   * Create a ClassTag for a WeakTypeTag
   *
   * @tparam T type T
   * @return ClassTag[T]
   */
  def classTagForWeakTypeTag[T: WeakTypeTag]: ClassTag[T] = {
    val t = weakTypeTag[T]
    ClassTag[T](t.mirror.runtimeClass(t.tpe))
  }

  /**
   * Return a TypeTag of typeTag[T].tpe.dealias (see [[TypeApi.dealias]])
   *
   * @param rtm  runtime mirror
   * @param ttag TypeTag[T]
   * @tparam T type T
   * @return TypeTag of typeTag[T].tpe.dealias
   */
  def dealiasedTypeTagForType[T](rtm: Mirror = runtimeMirror())
    (implicit ttag: TypeTag[T]): TypeTag[T] = typeTagForType[T](tpe = ttag.tpe.dealias, rtm = rtm)

  /**
   * Creates a Manifest[T] of a TypeTag[T]
   *
   * @tparam T type T with a TypeTag
   * @return Manifest[T]
   */
  def manifestForTypeTag[T: TypeTag]: Manifest[T] = {
    val t = typeTag[T]
    val mirror = t.mirror

    def toManifestRec(t: Type): Manifest[_] = {
      val clazz = ClassTag[T](mirror.runtimeClass(t)).runtimeClass
      if (t.typeArgs.length == 1) {
        val arg = toManifestRec(t.typeArgs.head)
        ManifestFactory.classType(clazz, arg)
      } else if (t.typeArgs.length > 1) {
        val args = t.typeArgs.map(x => toManifestRec(x))
        ManifestFactory.classType(clazz, args.head, args.tail: _*)
      } else {
        ManifestFactory.classType(clazz)
      }
    }

    toManifestRec(t.tpe).asInstanceOf[Manifest[T]]
  }

  /**
   * Given a Class this will return the runtime mirror and class mirror
   *
   * @param klazz       instance class
   * @param classLoader class loader to use
   * @return tuple of runtime mirror and class mirror
   */
  private def mirrors(klazz: Class[_], classLoader: ClassLoader): (Mirror, ClassMirror) = {
    val rtm = runtimeMirror()
    val klazzSymbol = rtm.classSymbol(klazz)
    rtm -> rtm.reflectClass(klazzSymbol)
  }

  /**
   * Reflects the best matching ctor that matches given ctor args extract function
   *
   * @return ctor method mirror with extracted args
   */
  private def bestCtor(
    klazz: Class[_],
    classType: Type,
    classMirror: ClassMirror,
    ctorArgs: (String, Symbol) => util.Try[Any]
  ): (MethodMirror, List[(String, Any)]) = {
    val ctorsWithParams = allCtorsWithParams(classType, classMirror)
    val ctorCandidates = ctorsWithParams.map { case (ctorMethod, ctorParamLists) =>
      util.Try(ctorMethod -> extractParams(klazz, ctorParamLists, ctorArgs))
    }
    ctorCandidates.partition(_.isSuccess) match {
      case (Success(ctor) :: _, _) => ctor
      case (_, Failure(error) :: _) => throw error
      case _ => throw new RuntimeException(s"No constructors were found for type ${klazz.getName}")
    }
  }

  /**
   * Cycles through the parameter list and uses the paramGetter lambda function to extract
   * the param value by name. If a param extraction has failed a RuntimeException will be
   * thrown
   *
   * @param klazz instance class
   * @param paramLists  param list to try and get values for
   * @param paramGetter lambda function to extract the param value by name.
   * @return List of tuples with the param name and it's value
   */
  private def extractParams
  (
    klazz: Class[_],
    paramLists: List[List[Symbol]],
    paramGetter: (String, Symbol) => util.Try[Any]
  ): List[(String, Any)] = {
    val paramValues = for {
      paramList <- paramLists
      param <- paramList
      paramName = termNameStr(param.name.toTermName)
    } yield paramName -> paramGetter(paramName, param)

    paramValues.collectFirst { case (paramName, Failure(error)) =>
      throw new RuntimeException(
        s"Failed to extract value for param '$paramName' " +
          s"for an instance of type '${klazz.getName}' due to: ${error.getMessage}",
        error
      )
    }
    paramValues.collect { case (n, Success(v)) => n -> v }
  }

  /**
   * Reflects and returns all type ctors + ctor params sorted by params length (desc)
   */
  private def allCtorsWithParams(
    classType: Type, classMirror: ClassMirror
  ): List[(MethodMirror, List[List[Symbol]])] = {
    // reflect all type constructors
    val ctors =
      classType.decl(runtimeUniverse.termNames.CONSTRUCTOR)
        .asTerm.alternatives.filter(_.isConstructor).map(_.asMethod)

    // reflect params for all constructors and sort by the number of params (desc)
    ctors
      .map(ctor => classMirror.reflectConstructor(ctor) -> ctor.paramLists)
      .sortBy(-_._2.map(_.length).sum)
  }

  private def termNameStr(termName: TermName): String = termName.decodedName.toString.trim

}
