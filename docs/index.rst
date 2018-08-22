.. TransmogrifAI master file, created by
   sphinx-quickstart on Tue Aug 21 21:25:31 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TransmogrifAI
=========================================

TransmogrifAI (pronounced trăns-mŏgˈrə-fī) is an **AutoML** library written in Scala that runs on top of Spark. It was developed with a focus on enhancing machine learning **developer productivity** through **machine learning automation**, and an API that enforces **compile-time type-safety**, **modularity** and **reuse**.

Use TransmogrifAI if you need a machine learning library to:

* Rapidly train good quality machine learnt models with **minimal hand tuning**
* Build modular, reusable, strongly typed machine learning workflows


-----


Guiding Principles
######################################

**Automation**:
TransmogrifAI has numerous Estimators (algorithms) that make use of TransmogrifAI feature types to automate feature engineering, feature selection, and model selection. Using these together with TransmogrifAI code-gen tools, the time taken to develop a very good model can be reduced from **several weeks** to **a couple of hours**!

**Modularity and reuse**:
TransmogrifAI enforces a strict separation between ML workflow definitions and data manipulation, ensuring that code written using TransmogrifAI is inherently modular and reusable.

**Compile-time type safety**:
Machine learning workflows built using TransmogrifAI are strongly typed. This means developers get to enjoy the many benefits of compile-time type safety, including code completion during development and fewer runtime errors. Workflows no longer fail several hours into model training because you tried to divide two strings!

**Transparency**:
The type-safe nature of TransmogrifAI ensures increased transparency around inputs and outputs at every stage of your machine learning workflow. This in turn greatly reduces the amount of tribal knowledge that inevitably tends to accumulate around any sufficiently complex machine learning workflow.

Motivation
######################################
*Building real life machine learning applications needs a fair amount of tribal knowledge and intuition. Coupled with the explosion of ML use cases in the world that need to be addressed, there is a need for tools that enable rapid prototyping and development of machine learning pipelines. We believe that automation is the key to making machine learning development truly scalable and accessible.*

For more information, read our `blogpost <https://engineering.salesforce.com/open-sourcing-transmogrifai-4e5d0e098da2/>`_!

Documentation
######################################
.. toctree::
   :maxdepth: 4

   installation/index
   examples/index
   abstractions/index
   automl-capabilities/index
   faq/index
   talks/index
   contributing/index
   developer-guide/index
   license/index



