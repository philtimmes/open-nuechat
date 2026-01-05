import { useState } from 'react';
import { GPTCategory } from './types';
import api from '../../lib/api';

interface Props {
  categories: GPTCategory[];
  setCategories: (categories: GPTCategory[]) => void;
  loading: boolean;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setSuccess: (success: string | null) => void;
}

export default function CategoriesTab({ 
  categories, 
  setCategories, 
  loading, 
  setLoading,
  setError, 
  setSuccess 
}: Props) {
  const [showModal, setShowModal] = useState(false);
  const [editingCategory, setEditingCategory] = useState<GPTCategory | null>(null);
  const [form, setForm] = useState({
    value: '',
    label: '',
    icon: '📁',
    description: '',
    sort_order: 0,
  });

  const resetForm = () => {
    setForm({
      value: '',
      label: '',
      icon: '📁',
      description: '',
      sort_order: 0,
    });
    setEditingCategory(null);
  };

  const startEdit = (cat: GPTCategory) => {
    setEditingCategory(cat);
    setForm({
      value: cat.value,
      label: cat.label,
      icon: cat.icon,
      description: cat.description || '',
      sort_order: cat.sort_order,
    });
    setShowModal(true);
  };

  const saveCategory = async () => {
    setLoading(true);
    try {
      if (editingCategory) {
        await api.patch(`/assistants/categories/${editingCategory.id}`, {
          label: form.label,
          icon: form.icon,
          description: form.description || null,
          sort_order: form.sort_order,
        });
        setSuccess('Category updated');
      } else {
        await api.post('/assistants/categories', form);
        setSuccess('Category created');
      }
      // Refresh
      const res = await api.get('/assistants/categories', { params: { include_inactive: true } });
      setCategories(res.data || []);
      setShowModal(false);
      resetForm();
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setError(error.response?.data?.detail || 'Failed to save category');
    } finally {
      setLoading(false);
    }
  };

  const toggleActive = async (cat: GPTCategory) => {
    try {
      await api.patch(`/assistants/categories/${cat.id}`, {
        is_active: !cat.is_active,
      });
      setCategories(categories.map(c => 
        c.id === cat.id ? { ...c, is_active: !c.is_active } : c
      ));
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setError(error.response?.data?.detail || 'Failed to toggle category');
    }
  };

  const deleteCategory = async (cat: GPTCategory) => {
    if (!confirm(`Delete category "${cat.label}"? GPTs using this category will be moved to "general".`)) return;
    
    try {
      await api.delete(`/assistants/categories/${cat.id}`);
      setCategories(categories.filter(c => c.id !== cat.id));
      setSuccess('Category deleted');
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setError(error.response?.data?.detail || 'Failed to delete category');
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-[var(--color-surface)] rounded-xl p-6 border border-[var(--color-border)]">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-[var(--color-text)]">GPT Categories</h3>
            <p className="text-sm text-[var(--color-text-secondary)]">
              Manage categories for organizing Custom GPTs in the marketplace
            </p>
          </div>
          <button
            onClick={() => {
              resetForm();
              setShowModal(true);
            }}
            className="px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90"
          >
            + Add Category
          </button>
        </div>
        
        {loading ? (
          <div className="text-[var(--color-text-secondary)]">Loading...</div>
        ) : categories.length === 0 ? (
          <div className="text-center py-8 text-[var(--color-text-secondary)]">
            No categories yet. Click "Add Category" to create one.
          </div>
        ) : (
          <div className="space-y-2">
            {categories.map(cat => (
              <div
                key={cat.id}
                className={`flex items-center justify-between p-4 rounded-lg transition-colors ${
                  cat.is_active 
                    ? 'bg-[var(--color-background)]' 
                    : 'bg-[var(--color-background)]/50 opacity-60'
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className="text-2xl w-10 text-center">{cat.icon}</span>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-[var(--color-text)]">{cat.label}</span>
                      <code className="text-xs px-2 py-0.5 rounded bg-[var(--color-surface)] text-[var(--color-text-secondary)]">
                        {cat.value}
                      </code>
                      {!cat.is_active && (
                        <span className="text-xs px-2 py-0.5 rounded bg-yellow-500/20 text-yellow-400">
                          Disabled
                        </span>
                      )}
                    </div>
                    {cat.description && (
                      <div className="text-sm text-[var(--color-text-secondary)]">{cat.description}</div>
                    )}
                  </div>
                </div>
                
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[var(--color-text-secondary)] mr-2">
                    Order: {cat.sort_order}
                  </span>
                  <button
                    onClick={() => startEdit(cat)}
                    className="px-3 py-1 text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] rounded hover:bg-[var(--color-border)]"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => toggleActive(cat)}
                    className={`px-3 py-1 text-sm rounded ${
                      cat.is_active
                        ? 'bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30'
                        : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                    }`}
                  >
                    {cat.is_active ? 'Disable' : 'Enable'}
                  </button>
                  {cat.value !== 'general' && (
                    <button
                      onClick={() => deleteCategory(cat)}
                      className="px-3 py-1 text-sm bg-red-500/20 text-red-400 rounded hover:bg-red-500/30"
                    >
                      Delete
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Category Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-[var(--color-surface)] rounded-xl p-6 w-full max-w-md border border-[var(--color-border)]">
            <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">
              {editingCategory ? 'Edit Category' : 'New Category'}
            </h3>
            
            <div className="space-y-4">
              {!editingCategory && (
                <div>
                  <label className="block text-sm text-[var(--color-text-secondary)] mb-1">
                    Value (slug)
                    <span className="text-red-400 ml-1">*</span>
                  </label>
                  <input
                    type="text"
                    value={form.value}
                    onChange={(e) => setForm({ ...form, value: e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, '') })}
                    placeholder="e.g. marketing"
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                  />
                  <p className="text-xs text-[var(--color-text-secondary)] mt-1">Lowercase letters, numbers, hyphens only</p>
                </div>
              )}
              
              <div>
                <label className="block text-sm text-[var(--color-text-secondary)] mb-1">
                  Label
                  <span className="text-red-400 ml-1">*</span>
                </label>
                <input
                  type="text"
                  value={form.label}
                  onChange={(e) => setForm({ ...form, label: e.target.value })}
                  placeholder="e.g. Marketing"
                  className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                />
              </div>
              
              <div>
                <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Icon (emoji)</label>
                <input
                  type="text"
                  value={form.icon}
                  onChange={(e) => setForm({ ...form, icon: e.target.value })}
                  className="w-24 px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-2xl text-center"
                />
              </div>
              
              <div>
                <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Description</label>
                <input
                  type="text"
                  value={form.description}
                  onChange={(e) => setForm({ ...form, description: e.target.value })}
                  placeholder="Optional description"
                  className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                />
              </div>
              
              <div>
                <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Sort Order</label>
                <input
                  type="number"
                  value={form.sort_order}
                  onChange={(e) => setForm({ ...form, sort_order: parseInt(e.target.value) || 0 })}
                  className="w-24 px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                />
                <p className="text-xs text-[var(--color-text-secondary)] mt-1">Lower numbers appear first</p>
              </div>
            </div>
            
            <div className="flex justify-end gap-2 mt-6">
              <button
                onClick={() => {
                  setShowModal(false);
                  resetForm();
                }}
                className="px-4 py-2 text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
              >
                Cancel
              </button>
              <button
                onClick={saveCategory}
                disabled={!form.value || !form.label || loading}
                className="px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 disabled:opacity-50"
              >
                {loading ? 'Saving...' : editingCategory ? 'Update' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
