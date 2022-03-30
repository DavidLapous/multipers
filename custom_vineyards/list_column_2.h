/*    This file is part of the MMA Library - https://gitlab.inria.fr/dloiseau/multipers - which is released under MIT.
 *    See file LICENSE for full license details.
 *    Author(s):       Hannah Schreiber
 *
 *    Copyright (C) 2022 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

#ifndef LISTCOLUMN_H
#define LISTCOLUMN_H

#include <iostream>
#include <list>
#include <unordered_set>

#include "utilities.h"

namespace Vineyard {

class List_column
{
public:
    List_column();
    List_column(boundary_type& boundary);
    List_column(List_column& column);
    List_column(List_column&& column) noexcept;

    void get_content(boundary_type& container);
    bool contains(unsigned int value) const;
    bool is_empty();
    dimension_type get_dimension() const;
    int get_pivot();
    void clear();
    void clear(unsigned int value);
    void reorder(std::vector<index>& valueMap);
    void add(List_column& column);

    List_column& operator=(List_column other);

    friend void swap(List_column& col1, List_column& col2);

private:
    int dim_;
    std::list<unsigned int> column_;
    std::unordered_set<unsigned int> erasedValues_;

    void _cleanValues();
};

inline List_column::List_column() : dim_(0)
{}

inline List_column::List_column(boundary_type &boundary)
    : dim_(boundary.size() == 0 ? 0 : boundary.size() - 1),
      column_(boundary.begin(), boundary.end())
{}

inline List_column::List_column(List_column &column)
    : dim_(column.dim_),
      column_(column.column_),
      erasedValues_(column.erasedValues_)
{}

inline List_column::List_column(List_column &&column) noexcept
    : dim_(std::exchange(column.dim_, 0)),
      column_(std::move(column.column_)),
      erasedValues_(std::move(column.erasedValues_))
{}

inline void List_column::get_content(boundary_type &container)
{
    _cleanValues();
    std::copy(column_.begin(), column_.end(), std::back_inserter(container));
}

inline bool List_column::contains(unsigned int value) const
{
    if (erasedValues_.find(value) != erasedValues_.end()) return false;

    for (unsigned int v : column_){
        if (v == value) return true;
    }
    return false;
}

inline bool List_column::is_empty()
{
    _cleanValues();
    return column_.empty();
}

inline dimension_type List_column::get_dimension() const
{
    return dim_;
}

inline int List_column::get_pivot()
{
    while (!column_.empty() &&
           erasedValues_.find(column_.back()) != erasedValues_.end()) {
        erasedValues_.erase(column_.back());
        column_.pop_back();
    }

    if (column_.empty()) return -1;

    return column_.back();
}

inline void List_column::clear()
{
    column_.clear();
    erasedValues_.clear();
}

inline void List_column::clear(unsigned int value)
{
    erasedValues_.insert(value);
}

inline void List_column::reorder(std::vector<index> &valueMap)
{
    std::list<unsigned int>::iterator it = column_.begin();
    while (it != column_.end()) {
        unsigned int erased = erasedValues_.erase(*it);
        if (erased > 0)
            it = column_.erase(it);
        else {
            *it = valueMap.at(*it);
            it++;
        }
    }
    column_.sort();
    erasedValues_.clear();
}

inline void List_column::add(List_column &column)
{
    if (column.is_empty()) return;
    if (column_.empty()){
        std::copy(column.column_.begin(), column.column_.end(), std::back_inserter(column_));
        return;
    }

    std::list<unsigned int>::iterator itToAdd = column.column_.begin();
    std::list<unsigned int>::iterator itTarget = column_.begin();
    unsigned int valToAdd = *itToAdd;
    unsigned int valTarget = *itTarget;

    while (itToAdd != column.column_.end() && itTarget != column_.end())
    {
        while (itToAdd != column.column_.end() &&
               column.erasedValues_.find(valToAdd) != column.erasedValues_.end()) {
            column.column_.erase(itToAdd++);
            column.erasedValues_.erase(valToAdd);
            valToAdd = *itToAdd;
        }

        while (itTarget != column_.end() &&
               erasedValues_.find(valTarget) != erasedValues_.end()) {
            column_.erase(itTarget++);
            valTarget = *itTarget;
        }

        if (itToAdd != column.column_.end() && itTarget != column_.end()){
            if (valToAdd == valTarget){
                column_.erase(itTarget++);
                itToAdd++;
            } else if (valToAdd < valTarget){
                column_.insert(itTarget, valToAdd);
                itToAdd++;
            } else {
                itTarget++;
            }
        }

        valToAdd = *itToAdd;
        valTarget = *itTarget;
    }

    while (itToAdd != column.column_.end()){
        valToAdd = *itToAdd;

        while (itToAdd != column.column_.end() &&
               column.erasedValues_.find(valToAdd) != column.erasedValues_.end()) {
            column.column_.erase(itToAdd++);
            column.erasedValues_.erase(valToAdd);
            valToAdd = *itToAdd;
        }

        if (itToAdd != column.column_.end()){
            column_.push_back(valToAdd);
            itToAdd++;
        }
    }

    erasedValues_.clear();
}

inline List_column &List_column::operator=(List_column other)
{
    std::swap(dim_, other.dim_);
    std::swap(column_, other.column_);
    std::swap(erasedValues_, other.erasedValues_);
    return *this;
}

inline void List_column::_cleanValues()
{
    std::list<unsigned int>::iterator it = column_.begin();
    while (it != column_.end() && !erasedValues_.empty()) {
        unsigned int erased = erasedValues_.erase(*it);
        if (erased > 0)
            it = column_.erase(it);
        else
            it++;
    }
    erasedValues_.clear();
}

inline void swap(List_column& col1, List_column& col2)
{
    std::swap(col1.dim_, col2.dim_);
    col1.column_.swap(col2.column_);
    std::swap(col1.erasedValues_, col2.erasedValues_);
}

}   //namespace Vineyard

#endif // LISTCOLUMN_H
