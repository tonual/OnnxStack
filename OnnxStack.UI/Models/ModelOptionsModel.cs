﻿using OnnxStack.StableDiffusion.Common;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Models
{
    public class ModelOptionsModel : INotifyPropertyChanged
    {
        private string _name;
        private bool _isLoaded;
        private bool _isLoading;
        private bool _isEnabled;

        public string Name
        {
            get { return _name; }
            set { _name = value; NotifyPropertyChanged(); }
        }

        public bool IsLoaded
        {
            get { return _isLoaded; }
            set { _isLoaded = value; NotifyPropertyChanged(); }
        }

        public bool IsLoading
        {
            get { return _isLoading; }
            set { _isLoading = value; NotifyPropertyChanged(); }
        }

        public bool IsEnabled
        {
            get { return _isEnabled; }
            set { _isEnabled = value; NotifyPropertyChanged(); }
        }

        public IModelOptions ModelOptions { get; set; }


        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }
        #endregion
    }
}
